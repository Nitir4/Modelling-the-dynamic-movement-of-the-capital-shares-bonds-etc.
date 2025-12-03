import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

import capital_model as model


def params(**overrides):
    return {'total_capital': 1., 'initial_stocks_fraction': .5,
            'initial_bonds_fraction': .3, 'risk_free_rate_type': 'constant',
            'risk_free_rate_baseline': .06, 'baseline_stock_return': .08,
            'baseline_bond_return': .04, **overrides}


def monthly(n=12, **rates):
    return pd.DataFrame({'risk_free_rate': .06, 'stock_return': .08, 'bond_return': .04, **rates},
                        index=pd.date_range('2020-01-31', periods=n, freq='ME'))


def test_generator_conserves_budget_and_points_inward_at_boundaries():
    p = params()
    for rates in [(0., 2., -1.), (.1, -.5, 3.), (.06, .08, .04)]:
        q = model.transfer_matrix(p, *rates)
        np.testing.assert_allclose(q.sum(axis=0), 0., atol=1e-15)
        assert (q-np.diag(np.diag(q)) >= 0).all()
        for state in np.eye(3):
            derivative = model.capital_flow_ode(0, state, p, *rates)
            assert (derivative[state == 0] >= 0).all()


def test_high_return_run_cannot_borrow_from_cash():
    p = params(initial_stocks_fraction=1., initial_bonds_fraction=0., baseline_stock_return=.3)
    _, frame = model.run_simulation(p, time_points=np.linspace(0, 30, 301))
    assert frame[['stocks', 'bonds', 'cash']].to_numpy().min() >= -1e-12
    np.testing.assert_allclose(frame[['stocks', 'bonds', 'cash']].sum(axis=1), 1., atol=2e-14)


def test_shocked_equilibrium_solves_the_same_return_functions():
    p = params()
    rf = .08
    state = model.steady_state(p, fixed_risk_free_rate=rf)
    residual = model.capital_flow_ode(0, state, p, rf, model.stock_return(rf, p), model.bond_return(rf, p))
    np.testing.assert_allclose(residual, 0, atol=1e-15)
    assert state.min() >= 0 and state.sum() == pytest.approx(1)
    assert p['risk_free_rate_baseline'] == .06


def test_jacobian_matches_independent_finite_differences():
    p, rates, state = params(), (.08, .078, .06), np.array([.4, .3])
    eps = 1e-6
    numeric = np.column_stack([(np.array(model.capital_flow_ode(0, state+eps*axis, p, *rates))-
                                np.array(model.capital_flow_ode(0, state-eps*axis, p, *rates)))/(2*eps)
                               for axis in np.eye(2)])
    j, eig, stable = model.jacobian_and_stability_at(*state, p, *rates)
    np.testing.assert_allclose(j, numeric, atol=1e-11)
    assert stable and (eig.real < 0).all()


def test_monthly_and_continuous_use_identical_year_units():
    p = params()
    a = model.run_simulation_with_timeseries(p, monthly())
    _, b = model.run_simulation(p, time_points=a.time.to_numpy())
    np.testing.assert_allclose(a[['stocks', 'bonds', 'cash']], b[['stocks', 'bonds', 'cash']], atol=1e-14)
    assert not a.time.duplicated().any()
    assert a.time.iloc[-1] == pytest.approx(1.)
    assert a.date.iloc[0] == pd.Timestamp('2020-01-01')
    assert a.date.iloc[-1] == pd.Timestamp('2021-01-01')


def test_step_shock_composes_exact_solutions_even_without_sample_at_shock():
    p = params(risk_free_rate_type='step', risk_free_rate_shock_time=.35, risk_free_rate_shock_magnitude=.04)
    _, frame = model.run_simulation(p, time_points=[0., 1.])
    before = model.transfer_matrix(p, .06, .08, .04)
    after = model.transfer_matrix(p, .10, .076, .08)
    expected = expm(after*.65) @ expm(before*.35) @ np.array([.5, .3, .2])
    np.testing.assert_allclose(frame[['stocks', 'bonds', 'cash']].iloc[-1], expected, atol=1e-14)


def test_pure_decay_quarterly_integral_is_resolution_independent():
    p = params(stock_sensitivity_to_excess_return=0., bond_sensitivity_to_excess_return=0.,
               cash_to_stocks_baseline_flow=0., cash_to_bonds_baseline_flow=0.,
               stock_natural_decay_rate=.1, bond_natural_decay_rate=0.,
               initial_stocks_fraction=1., initial_bonds_fraction=0.)
    tables = []
    for resolution in [2, 10, 40]:
        frame = model.run_simulation_with_timeseries(p, monthly(), resolution)
        table, _ = model.generate_quarterly_summary(frame, monthly(), p)
        expected = np.exp(-.1*np.arange(1, 5)/4)-np.exp(-.1*np.arange(4)/4)
        np.testing.assert_allclose(table.delta_S, expected, atol=1e-14)
        np.testing.assert_allclose(table.total_flow_to_stocks, expected, atol=1e-14)
        assert table.quarter.tolist() == ['2020Q1', '2020Q2', '2020Q3', '2020Q4']
        tables.append(table)
    for table in tables[1:]:
        np.testing.assert_allclose(table.delta_S, tables[0].delta_S, atol=1e-14)


@pytest.mark.parametrize('column', ['rf', 'rs', 'rb'])
@pytest.mark.parametrize('location', ['leading', 'trailing', 'infinite'])
def test_all_missing_rate_columns_and_trailing_gaps_are_validated(column, location):
    raw = pd.DataFrame({'date': pd.date_range('2020-01-31', periods=5, freq='ME'), 'rf': .06, 'rs': .08, 'rb': .04})
    if location == 'leading': raw.loc[0, column] = np.nan
    elif location == 'trailing': raw.loc[2:, column] = np.nan
    else: raw.loc[2, column] = np.inf
    with pytest.raises(ValueError): model.preprocess_timeseries(raw, False, True)


def test_preprocessing_carries_last_whole_row_without_future_interpolation():
    raw = pd.DataFrame({'date': ['2020-01-31', '2020-02-10', '2020-02-29', '2020-03-31'],
                        'rf': [.01, .02, .03, .04], 'rs': [.1, .2, np.nan, .4], 'rb': [.05]*4})
    frame = model.preprocess_timeseries(raw, False, True)
    assert frame.stock_return.iloc[1] == .1  # January's observed rate, not February 10 or March
    assert frame.risk_free_rate.iloc[1] == .03
    assert frame.attrs['imputed_cells'] == 1


def test_effective_and_continuous_monthly_rates_convert_to_annual_units():
    raw = pd.DataFrame({'date': ['2020-01-31'], 'rf': [1.], 'rs': [2.], 'rb': [.5]})
    effective = model.preprocess_timeseries(raw, True, False, 'effective')
    np.testing.assert_allclose(effective.iloc[0], 12*np.log1p([.01, .02, .005]))
    continuous = model.preprocess_timeseries(raw, True, False)
    np.testing.assert_allclose(continuous.iloc[0], [.12, .24, .06])


def test_nonunique_equilibrium_is_reported_instead_of_fabricated():
    p = params(stock_sensitivity_to_excess_return=0., bond_sensitivity_to_excess_return=0.,
               cash_to_stocks_baseline_flow=0., cash_to_bonds_baseline_flow=0.,
               stock_natural_decay_rate=0., bond_natural_decay_rate=0.)
    assert model.steady_state_one_month(p, .06, .08, .04) is None
    with pytest.raises(RuntimeError, match='unique'): model.steady_state(p)


def test_diagnostics_do_not_clamp_or_hide_actual_balance_errors():
    frame = pd.DataFrame({'time': [0.], 'stocks': [.5], 'bonds': [.3], 'cash': [.21]})
    diagnostics = model.add_diagnostics(frame, params())
    assert diagnostics['max_conservation_error'] == pytest.approx(.01)
    assert frame.cash.iloc[0] == .21
    assert not diagnostics['clamping_occurred']


@pytest.mark.parametrize('overrides', [{'initial_stocks_fraction': .8, 'initial_bonds_fraction': .4},
                                        {'total_capital': 0.}, {'cash_to_stocks_baseline_flow': -.1}])
def test_invalid_allocations_and_coefficients_fail(overrides):
    with pytest.raises(ValueError): model.run_simulation(params(**overrides), time_points=[0., 1.])


def test_app_imports_the_corrected_core_and_embedded_analytic_checks_pass():
    import streamlit_main as app
    assert app.capital_flow_ode is model.capital_flow_ode
    assert all(result['pass'] for result in app.run_unit_tests(app.DEFAULT_PARAMS).values())
