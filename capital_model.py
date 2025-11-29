"""Funded allocation transfers, model v2. Time and coefficients use years.

Returns are continuous annualized incentives, not credited portfolio gains.
The three balances evolve independently under a conservative transfer generator.
"""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def risk_free_rate(time, params):
    baseline = params.get('risk_free_rate_baseline', .06)
    if not np.isfinite([time, baseline, params.get('risk_free_rate_shock_magnitude', .02)]).all():
        raise ValueError('Times and rate parameters must be finite.')
    kind = params.get('risk_free_rate_type', 'constant')
    if kind == 'constant':
        return baseline
    if kind == 'step':
        if not np.isfinite(params.get('risk_free_rate_shock_time', 20.)):
            raise ValueError('Shock time must be finite.')
        return baseline + (params.get('risk_free_rate_shock_magnitude', .02)
                           if time >= params.get('risk_free_rate_shock_time', 20.) else 0.)
    if kind == 'sin':
        period = params.get('risk_free_rate_sine_period', 60.)
        if not np.isfinite(period) or period <= 0:
            raise ValueError('Sine period must be positive years.')
        return baseline + params.get('risk_free_rate_shock_magnitude', .02)*np.sin(2*np.pi*time/period)
    raise ValueError(f'Unknown rate type: {kind}')


def stock_return(rf, params):
    return params.get('baseline_stock_return', .08) + params.get('lambda_s', -.1)*(rf-params.get('risk_free_rate_baseline', .06))


def bond_return(rf, params):
    """Illustrative constant expected excess-return spread; no duration pricing."""
    return rf + params.get('baseline_bond_return', .04)-params.get('risk_free_rate_baseline', .06)


def transfer_rates(params, rf, rs, rb):
    names = ['stock_sensitivity_to_excess_return', 'bond_sensitivity_to_excess_return',
             'cash_to_stocks_baseline_flow', 'cash_to_bonds_baseline_flow',
             'stock_natural_decay_rate', 'bond_natural_decay_rate']
    defaults = [.75, .4, .03, .03, .015, .01]
    values = np.array([params.get(n, d) for n, d in zip(names, defaults)], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.isfinite(np.asarray([rf, rs, rb], dtype=float)).all():
        raise ValueError('Use finite rates and nonnegative sensitivities/transfer coefficients.')
    a_s, a_b, b_s, b_b, g_s, g_b = values
    e_s, e_b = a_s*(rs-rf), a_b*(rb-rf)
    return b_s+max(e_s, 0.), b_b+max(e_b, 0.), g_s+max(-e_s, 0.), g_b+max(-e_b, 0.)


def transfer_matrix(params, rf, rs, rb):
    u_s, u_b, d_s, d_b = transfer_rates(params, rf, rs, rb)
    # Columns sum to zero; off-diagonal entries are nonnegative.
    return np.array([[-d_s, 0., u_s], [0., -d_b, u_b], [d_s, d_b, -u_s-u_b]])


def _rates(time, params):
    rf = risk_free_rate(time, params)
    return rf, stock_return(rf, params), bond_return(rf, params)


def capital_flow_ode(time, state, params, fixed_rf=None, fixed_rs=None, fixed_rb=None):
    rates = _rates(time, params) if fixed_rf is None else (fixed_rf, fixed_rs, fixed_rb)
    state = np.asarray(state, dtype=float)
    if state.shape == (2,):
        full = np.r_[state, params.get('total_capital', 1.)-state.sum()]
        return (transfer_matrix(params, *rates) @ full)[:2]
    if state.shape != (3,):
        raise ValueError('State must contain S, B or S, B, C.')
    return transfer_matrix(params, *rates) @ state


def _initial(params, initial_state=None):
    k = params.get('total_capital', 1.)
    if not np.isfinite(k) or k <= 0:
        raise ValueError('Total capital must be positive.')
    state = np.array(initial_state if initial_state is not None else
                     [k*params.get('initial_stocks_fraction', .5), k*params.get('initial_bonds_fraction', .3)], dtype=float)
    if state.shape == (2,):
        state = np.r_[state, k-state.sum()]
    if state.shape != (3,) or not np.isfinite(state).all() or (state < 0).any() or not np.isclose(state.sum(), k, rtol=1e-12, atol=1e-12*k):
        raise ValueError('Initial allocations must be nonnegative and sum to K.')
    return state


def preprocess_timeseries(df, is_percent, is_annual, rate_convention='continuous'):
    """Last whole row per month, past-only carry for at most two missing months.

    Annual inputs are continuous annualized rates; monthly inputs are continuous
    monthly rates multiplied by 12. Effective returns need log1p conversion first.
    Monthly labels describe historical forcing for that month, not forecasts.
    """
    if not {'date', 'rf', 'rs', 'rb'}.issubset(df.columns) or df.empty:
        raise ValueError('A nonempty CSV needs date, rf, rs, rb.')
    df = df[['date', 'rf', 'rs', 'rb']].copy()
    df['date'] = pd.to_datetime(df.date, errors='raise')
    if df.date.isna().any() or df.date.duplicated().any():
        raise ValueError('Dates must be present and unique.')
    df = df.sort_values('date')
    periods = df.date.dt.to_period('M')
    monthly = df.groupby(periods, sort=True).tail(1).copy()
    monthly.index = monthly.date.dt.to_period('M')
    monthly = monthly[['rf', 'rs', 'rb']].apply(pd.to_numeric, errors='raise')
    if np.isinf(monthly.to_numpy()).any():
        raise ValueError('Rates cannot be infinite.')
    monthly = monthly.reindex(pd.period_range(monthly.index.min(), monthly.index.max(), freq='M'))
    missing_count = int(monthly.isna().sum().sum())
    monthly = monthly.ffill(limit=2)
    if monthly.isna().any().any():
        raise ValueError('Every rate needs a prior observation; gaps may not exceed two months.')
    monthly *= 1./100 if is_percent else 1.
    if rate_convention == 'effective':
        if (monthly <= -1).any().any():
            raise ValueError('Effective returns must exceed -100 percent.')
        monthly = np.log1p(monthly)
    elif rate_convention != 'continuous':
        raise ValueError('Rate convention must be continuous or effective.')
    monthly *= 1. if is_annual else 12.
    monthly.index = monthly.index.to_timestamp('M')
    monthly.columns = ['risk_free_rate', 'stock_return', 'bond_return']
    monthly.attrs.update(rate_units='continuous annualized', imputed_cells=missing_count)
    return monthly


def _frame(times, states, rates):
    return pd.DataFrame(dict(time=times, stocks=states[:, 0], bonds=states[:, 1], cash=states[:, 2],
                             risk_free_rate=rates[:, 0], stock_return=rates[:, 1], bond_return=rates[:, 2]))


def run_simulation_with_timeseries(params, timeseries_df, steps_per_month=10):
    if isinstance(steps_per_month, bool) or not isinstance(steps_per_month, (int, np.integer)) or steps_per_month < 2:
        raise ValueError('Use at least two points per month.')
    required = ['risk_free_rate', 'stock_return', 'bond_return']
    if timeseries_df.empty or not isinstance(timeseries_df.index, pd.DatetimeIndex):
        raise ValueError('Use a nonempty monthly DatetimeIndex.')
    periods = timeseries_df.index.to_period('M')
    if not periods.equals(pd.period_range(periods[0], periods=len(periods), freq='M')):
        raise ValueError('Monthly observations must be unique, sorted and consecutive.')
    if not np.isfinite(timeseries_df[required].to_numpy()).all():
        raise ValueError('All forcing rates must be finite and annualized.')
    state = _initial(params)
    frames = []
    for i, (period, rate) in enumerate(zip(periods, timeseries_df[required].to_numpy())):
        q = transfer_matrix(params, *rate)
        local_times = np.linspace(0, 1./12, steps_per_month)
        states = np.array([expm(q*t) @ state for t in local_times])
        derivative = states @ q.T
        frame = _frame(i/12+local_times, states, np.tile(rate, (steps_per_month, 1)))
        frame['dS_dt'], frame['dB_dt'] = derivative[:, 0], derivative[:, 1]
        # Each reported increment belongs to the interval ending at this row.
        frame['interval_delta_S'] = np.r_[0., np.diff(states[:, 0])]
        frame['interval_delta_B'] = np.r_[0., np.diff(states[:, 1])]
        frame['month'] = str(period)
        start, end = period.start_time, (period+1).start_time
        frame['date'] = [start+(end-start)*(t*12) for t in local_times]
        frames.append(frame if i == 0 else frame.iloc[1:])
        state = states[-1]
    return pd.concat(frames, ignore_index=True)


def run_simulation(params, initial_state=None, time_points=None):
    times = np.asarray(time_points if time_points is not None else np.linspace(0., params.get('simulation_time_horizon', 200.), 1001), dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or times[0] < 0 or (np.diff(times) <= 0).any():
        raise ValueError('Times must be finite, nonnegative, strictly increasing years.')
    state = _initial(params, initial_state)
    _rates(times[0], params)  # validate on integration below
    cuts = [times[0], times[-1]]
    if params.get('risk_free_rate_type') == 'step':
        shock = params.get('risk_free_rate_shock_time', 20.)
        if not np.isfinite(shock):
            raise ValueError('Shock time must be finite.')
        if times[0] < shock < times[-1]:
            cuts.insert(1, shock)
    states = np.empty((len(times), 3))
    for a, b in zip(cuts[:-1], cuts[1:]):
        mask = (times >= a) & (times <= b)
        if params.get('risk_free_rate_type', 'constant') in ('step', 'constant'):
            q = transfer_matrix(params, *_rates((a+b)/2, params))
            states[mask] = [expm(q*(t-a)) @ state for t in times[mask]]
            state = expm(q*(b-a)) @ state
        else:
            sol = solve_ivp(lambda t, y: capital_flow_ode(t, y, params), (a, b), state,
                            dense_output=True, atol=1e-11, rtol=1e-9,
                            max_step=params.get('risk_free_rate_sine_period', 60.)/50)
            if not sol.success:
                raise RuntimeError(sol.message)
            states[mask] = sol.sol(times[mask]).T
            state = sol.y[:, -1]
    frame = _frame(times, states, np.array([_rates(t, params) for t in times]))
    return SimpleNamespace(t=times, y=states.T, success=True), frame


def jacobian_and_stability_at(stocks_value, bonds_value, params, rf, rs, rb):
    u_s, u_b, d_s, d_b = transfer_rates(params, rf, rs, rb)
    j = np.array([[-d_s-u_s, -u_s], [-u_b, -d_b-u_b]])
    eig = np.linalg.eigvals(j)
    return j, eig, bool((eig.real < -1e-12).all())


def steady_state_one_month(params, rf, rs, rb, initial_guess=None):
    """Unique feasible fixed-rate equilibrium; None for nonunique equilibria."""
    _initial(params)
    k = params.get('total_capital', 1.)
    q = transfer_matrix(params, rf, rs, rb)
    matrix, rhs = q.copy(), np.array([0., 0., k])
    matrix[-1] = 1.
    try:
        state = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(state).all() or state.min() < -1e-10*k or np.max(np.abs(q @ state)) > 1e-10*k:
        return None
    return tuple(state)


def steady_state(params, fixed_risk_free_rate=None, initial_guess=None):
    rf = params.get('risk_free_rate_baseline', .06) if fixed_risk_free_rate is None else fixed_risk_free_rate
    state = steady_state_one_month(params, rf, stock_return(rf, params), bond_return(rf, params))
    if state is None:
        raise RuntimeError('No unique feasible equilibrium; check zero transfer rates.')
    return np.array(state)


def add_diagnostics(results_df, params):
    """Measure independently evolved balance sums; do not alter allocations."""
    error = results_df[['stocks', 'bonds', 'cash']].sum(axis=1)-params.get('total_capital', 1.)
    negative = (results_df[['stocks', 'bonds', 'cash']] < -1e-8*params.get('total_capital', 1.)).any(axis=1)
    results_df['conservation_error'] = error
    results_df['bankruptcy_flag'] = negative
    results_df['clamped_flag'] = False
    return dict(conservation_mse=float((error**2).mean()), max_conservation_error=float(error.abs().max()),
                bankruptcy_occurred=bool(negative.any()),
                first_bankruptcy_time=float(results_df.loc[negative, 'time'].iloc[0]) if negative.any() else None,
                clamping_occurred=False)


def generate_quarterly_summary(results_df, timeseries_df, params):
    """Exact net-flow integral via balance increments, calendar-month grouping."""
    if 'month' not in results_df:
        raise ValueError('Quarterly summaries require monthly simulation output.')
    data = results_df.copy()
    data['quarter'] = pd.PeriodIndex(data.month, freq='M').asfreq('Q')
    rows, descriptions = [], []
    for quarter, group in data.groupby('quarter', sort=True):
        months = pd.PeriodIndex(group.month.unique(), freq='M')
        forcing = timeseries_df.loc[timeseries_df.index.to_period('M').isin(months)]
        delta_s, delta_b = group.interval_delta_S.sum(), group.interval_delta_B.sum()
        mid = group.iloc[len(group)//2]
        _, _, stable = jacobian_and_stability_at(mid.stocks, mid.bonds, params, mid.risk_free_rate, mid.stock_return, mid.bond_return)
        rows.append(dict(quarter=str(quarter), start_date=months[0].start_time.strftime('%Y-%m-%d'),
                         end_date=(months[-1]+1).start_time.strftime('%Y-%m-%d'), delta_S=delta_s, delta_B=delta_b,
                         avg_rf=forcing.risk_free_rate.mean(), avg_rs=forcing.stock_return.mean(), avg_rb=forcing.bond_return.mean(),
                         total_flow_to_stocks=delta_s, stability='Frozen-rate stable' if stable else 'Frozen-rate non-asymptotic'))
        descriptions.append(f'{quarter}: net stock transfer {delta_s/params.get("total_capital", 1.):.3%} of K. '
                            'Balance change includes every funded inflow and outflow; frozen-rate eigenvalues do not assess the entire varying-rate path.')
    return pd.DataFrame(rows), descriptions
