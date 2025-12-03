#!/usr/bin/env python3
"""
capital_flow_streamlit_app.py

Enhanced Streamlit web app for Capital Flow Model with time-series integration,
diagnostics, stability analysis, and quarterly reporting.

Run with: streamlit run capital_flow_streamlit_app.py

Sample CSV format for time-series upload:
date,rf,rs,rb
2020-01-31,6.50,12.0,7.20
2020-02-29,6.50,-8.0,7.10
2020-03-31,6.00,15.0,6.80
...

CSV columns:
- date: Any parseable date format (YYYY-MM-DD, MM/DD/YYYY, etc.)
- rf: Risk-free rate (repo rate)
- rs: Stock/equity return
- rb: Bond yield/return

Use checkboxes to indicate:
- Whether rates are in percent (will divide by 100)
- Whether rates are annual (will divide by 12 for monthly)
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import root
from numpy.linalg import eigvals
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime


# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

DEFAULT_PARAMS = {
    "stock_sensitivity_to_excess_return": 0.75,      # α_s
    "bond_sensitivity_to_excess_return": 0.4,       # α_b
    "cash_to_stocks_baseline_flow": 0.03,           # β_s
    "cash_to_bonds_baseline_flow": 0.03,            # β_b
    "stock_natural_decay_rate": 0.015,              # γ_s
    "bond_natural_decay_rate": 0.01,                # γ_b
    
    "baseline_stock_return": 0.08,                  # r_s^0
    "baseline_bond_return": 0.04,                   # r_b^0
    "lambda_s": -0.10,                              # sensitivity of stocks to rf
    "bond_duration_years": 5.0,
    
    "total_capital": 1.0,                           # K
    
    "risk_free_rate_baseline": 0.06,                # r_f^0
    "risk_free_rate_shock_magnitude": 0.02,
    "risk_free_rate_shock_time": 20.0,
    "risk_free_rate_type": "step",
    "risk_free_rate_sine_period": 60.0,
    "simulation_time_horizon": 200.0,
    
    "initial_stocks_fraction": 0.5,
    "initial_bonds_fraction": 0.3,
}


# ============================================================================
# CORE ODE MODEL (UNCHANGED)
# ============================================================================

def risk_free_rate(time, params):
    """Calculate time-varying risk-free rate."""
    rate_type = params.get("risk_free_rate_type", "constant")
    baseline_rate = params.get("risk_free_rate_baseline", 0.04)
    
    if rate_type == "constant":
        return baseline_rate
    elif rate_type == "step":
        shock_time = params.get("risk_free_rate_shock_time", 20.0)
        shock_magnitude = params.get("risk_free_rate_shock_magnitude", 0.02)
        if time >= shock_time:
            return baseline_rate + shock_magnitude
        else:
            return baseline_rate
    elif rate_type == "sin":
        period = params.get("risk_free_rate_sine_period", 60.0)
        amplitude = params.get("risk_free_rate_shock_magnitude", 0.01)
        return baseline_rate + amplitude * np.sin(2 * np.pi * time / period)
    else:
        raise ValueError(f"Unknown risk_free_rate_type: {rate_type}")


def stock_return(current_risk_free_rate, params):
    """Calculate expected stock return based on risk-free rate."""
    baseline_stock_return = params.get("baseline_stock_return", 0.08)
    stock_rate_sensitivity = params.get("lambda_s", -0.1)
    baseline_risk_free = params.get("risk_free_rate_baseline", 0.04)
    
    rate_delta = current_risk_free_rate - baseline_risk_free
    return baseline_stock_return + stock_rate_sensitivity * rate_delta


def bond_return(current_risk_free_rate, params):
    """Calculate bond return based on risk-free rate and spread."""
    baseline_bond_return = params.get("baseline_bond_return", 0.04)
    baseline_risk_free = params.get("risk_free_rate_baseline", 0.04)
    
    spread = baseline_bond_return - baseline_risk_free
    return current_risk_free_rate + spread


def capital_flow_ode(time, state, params, fixed_rf=None, fixed_rs=None, fixed_rb=None):
    """
    Define the system of ODEs for capital flows.
    
    state = [stocks_capital, bonds_capital]
    cash_capital = K - stocks_capital - bonds_capital
    
    dS/dt = α_s (r_s - r_f) S + β_s C - γ_s S
    dB/dt = α_b (r_b - r_f) B + β_b C - γ_b B
    
    Returns: [dS/dt, dB/dt]
    """
    stocks_capital, bonds_capital = state
    total_capital = params.get("total_capital", 1.0)
    cash_capital = total_capital - stocks_capital - bonds_capital
    
    # Use fixed rates if provided (for piecewise integration), otherwise compute
    if fixed_rf is not None:
        current_risk_free = fixed_rf
        current_stock_return = fixed_rs
        current_bond_return = fixed_rb
    else:
        current_risk_free = risk_free_rate(time, params)
        current_stock_return = stock_return(current_risk_free, params)
        current_bond_return = bond_return(current_risk_free, params)
    
    # Extract parameters
    alpha_s = params.get("stock_sensitivity_to_excess_return", 0.5)
    alpha_b = params.get("bond_sensitivity_to_excess_return", 0.4)
    beta_s = params.get("cash_to_stocks_baseline_flow", 0.02)
    beta_b = params.get("cash_to_bonds_baseline_flow", 0.03)
    gamma_s = params.get("stock_natural_decay_rate", 0.01)
    gamma_b = params.get("bond_natural_decay_rate", 0.01)
    
    stock_excess_return = current_stock_return - current_risk_free
    bond_excess_return = current_bond_return - current_risk_free
    
    dS_dt = (alpha_s * stock_excess_return * stocks_capital + 
             beta_s * cash_capital - 
             gamma_s * stocks_capital)
    
    dB_dt = (alpha_b * bond_excess_return * bonds_capital + 
             beta_b * cash_capital - 
             gamma_b * bonds_capital)
    
    return [dS_dt, dB_dt]


# ============================================================================
# A — TIME-SERIES UPLOADER & PREPROCESSING
# ============================================================================

def preprocess_timeseries(df, is_percent, is_annual):
    """
    Preprocess uploaded time-series CSV.
    
    Args:
        df: DataFrame with columns date, rf, rs, rb
        is_percent: If True, divide rates by 100
        is_annual: If True, divide rates by 12 to convert to monthly
    
    Returns:
        DataFrame indexed by month-end with columns: 
        risk_free_rate, stock_return, bond_return (in per-month decimal units)
    
    Raises:
        ValueError: If required columns missing or gaps > 2 months detected
    """
    # Validate required columns
    required_cols = ['date', 'rf', 'rs', 'rb']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Parse dates and sort
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Resample to month-end (take last observation in each month)
    df.set_index('date', inplace=True)
    df_monthly = df.resample('ME').last()
    
    # Reindex to full month-end range
    full_range = pd.date_range(start=df_monthly.index.min(), 
                                end=df_monthly.index.max(), 
                                freq='ME')
    df_reindexed = df_monthly.reindex(full_range)
    
    # Detect missing-month runs
    missing_mask = df_reindexed['rf'].isna()
    
    if missing_mask.any():
        # Find continuous runs of missing data
        missing_runs = []
        in_run = False
        run_start = None
        
        for idx, is_missing in enumerate(missing_mask):
            if is_missing and not in_run:
                in_run = True
                run_start = idx
            elif not is_missing and in_run:
                in_run = False
                run_length = idx - run_start
                missing_runs.append((run_start, idx - 1, run_length))
        
        # Check if any run > 2 months
        long_gaps = [run for run in missing_runs if run[2] > 2]
        if long_gaps:
            gap_details = []
            for start_idx, end_idx, length in long_gaps:
                start_date = full_range[start_idx].strftime('%Y-%m')
                end_date = full_range[end_idx].strftime('%Y-%m')
                gap_details.append(f"{start_date} to {end_date} ({length} months)")
            
            raise ValueError(
                f"Data contains gaps longer than 2 months. Cannot interpolate.\n"
                f"Gaps detected: {'; '.join(gap_details)}"
            )
        
        # Linear interpolation for gaps ≤ 2 months
        df_reindexed = df_reindexed.interpolate(method='linear', limit=2)
    
    # Convert rates
    conversion_factor = 1.0
    if is_percent:
        conversion_factor /= 100.0
    if is_annual:
        conversion_factor /= 12.0
    
    # Create clean output dataframe
    clean_df = pd.DataFrame({
        'risk_free_rate': df_reindexed['rf'] * conversion_factor,
        'stock_return': df_reindexed['rs'] * conversion_factor,
        'bond_return': df_reindexed['rb'] * conversion_factor
    }, index=full_range)
    
    return clean_df


# ============================================================================
# B — PIECEWISE MONTHLY INTEGRATION
# ============================================================================

def run_simulation_with_timeseries(params, timeseries_df, steps_per_month=10):
    """
    Run piecewise monthly integration using uploaded time-series data.
    
    Each month is treated as a unit interval with constant rates from that month.
    Integrates month-by-month, chaining initial conditions.
    
    Args:
        params: Parameter dictionary
        timeseries_df: DataFrame from preprocess_timeseries
        steps_per_month: Number of evaluation points per month
    
    Returns:
        DataFrame with columns: time, stocks, bonds, cash, risk_free_rate, 
        stock_return, bond_return, dS_dt, dB_dt
    """
    total_capital = params.get("total_capital", 1.0)
    initial_stocks = params.get("initial_stocks_fraction", 0.5) * total_capital
    initial_bonds = params.get("initial_bonds_fraction", 0.3) * total_capital
    
    current_state = [initial_stocks, initial_bonds]
    
    all_results = []
    num_months = len(timeseries_df)
    
    for month_idx in range(num_months):
        # Get rates for this month
        rf = timeseries_df.iloc[month_idx]['risk_free_rate']
        rs = timeseries_df.iloc[month_idx]['stock_return']
        rb = timeseries_df.iloc[month_idx]['bond_return']
        
        # Time span for this month: [month_idx, month_idx + 1]
        t_start = float(month_idx)
        t_end = float(month_idx + 1)
        t_eval = np.linspace(t_start, t_end, steps_per_month)
        
        # Define ODE with fixed rates for this month
        def ode_fixed_rates(t, y):
            return capital_flow_ode(t, y, params, 
                                   fixed_rf=rf, fixed_rs=rs, fixed_rb=rb)
        
        # Solve for this month
        sol = solve_ivp(
            fun=ode_fixed_rates,
            t_span=(t_start, t_end),
            y0=current_state,
            t_eval=t_eval,
            method='RK45',
            atol=1e-8,
            rtol=1e-6
        )
        
        if not sol.success:
            raise RuntimeError(f"ODE solver failed at month {month_idx}: {sol.message}")
        
        # Extract results
        S = sol.y[0]
        B = sol.y[1]
        C = total_capital - S - B
        
        # Compute derivatives at each point
        dS_dt = []
        dB_dt = []
        for i in range(len(sol.t)):
            derivatives = capital_flow_ode(
                sol.t[i], [S[i], B[i]], params,
                fixed_rf=rf, fixed_rs=rs, fixed_rb=rb
            )
            dS_dt.append(derivatives[0])
            dB_dt.append(derivatives[1])
        
        # Store month results
        month_df = pd.DataFrame({
            'time': sol.t,
            'stocks': S,
            'bonds': B,
            'cash': C,
            'risk_free_rate': rf,
            'stock_return': rs,
            'bond_return': rb,
            'dS_dt': dS_dt,
            'dB_dt': dB_dt
        })
        
        all_results.append(month_df)
        
        # Update initial condition for next month
        current_state = [S[-1], B[-1]]
    
    # Concatenate all months
    results_df = pd.concat(all_results, ignore_index=True)
    
    return results_df


# ============================================================================
# C — CONSERVATION DIAGNOSTICS & NEGATIVE HANDLING
# ============================================================================

def add_diagnostics(results_df, params):
    """
    Add conservation error, bankruptcy, and clamping diagnostics.
    
    Modifies results_df in place to add columns:
    - conservation_error: S + B + C - K
    - bankruptcy_flag: True if any S, B, or C < -1e-6
    - clamped_flag: True if any value in [-1e-6, 0) was clamped
    
    Returns:
        dict with summary diagnostics
    """
    K = params.get("total_capital", 1.0)
    
    # Conservation error
    results_df['conservation_error'] = (results_df['stocks'] + 
                                       results_df['bonds'] + 
                                       results_df['cash'] - K)
    
    conservation_mse = (results_df['conservation_error'] ** 2).mean()
    max_conservation_error = results_df['conservation_error'].abs().max()
    
    # Check for negative allocations
    severe_negative_S = (results_df['stocks'] < -1e-6).any()
    severe_negative_B = (results_df['bonds'] < -1e-6).any()
    severe_negative_C = (results_df['cash'] < -1e-6).any()
    
    bankruptcy_occurred = severe_negative_S or severe_negative_B or severe_negative_C
    
    # Find first bankruptcy time
    first_bankruptcy_time = None
    if bankruptcy_occurred:
        bankruptcy_mask = ((results_df['stocks'] < -1e-6) | 
                          (results_df['bonds'] < -1e-6) | 
                          (results_df['cash'] < -1e-6))
        if bankruptcy_mask.any():
            first_bankruptcy_time = results_df.loc[bankruptcy_mask, 'time'].iloc[0]
    
    results_df['bankruptcy_flag'] = bankruptcy_occurred
    
    # Clamping: values in [-1e-6, 0)
    tiny_negative_S = ((results_df['stocks'] >= -1e-6) & 
                       (results_df['stocks'] < 0)).any()
    tiny_negative_B = ((results_df['bonds'] >= -1e-6) & 
                       (results_df['bonds'] < 0)).any()
    tiny_negative_C = ((results_df['cash'] >= -1e-6) & 
                       (results_df['cash'] < 0)).any()
    
    clamping_occurred = tiny_negative_S or tiny_negative_B or tiny_negative_C
    results_df['clamped_flag'] = clamping_occurred
    
    # Clamp tiny negatives to zero for display
    if clamping_occurred:
        results_df.loc[results_df['stocks'] >= -1e-6, 'stocks'] = \
            results_df.loc[results_df['stocks'] >= -1e-6, 'stocks'].clip(lower=0)
        results_df.loc[results_df['bonds'] >= -1e-6, 'bonds'] = \
            results_df.loc[results_df['bonds'] >= -1e-6, 'bonds'].clip(lower=0)
        results_df.loc[results_df['cash'] >= -1e-6, 'cash'] = \
            results_df.loc[results_df['cash'] >= -1e-6, 'cash'].clip(lower=0)
    
    diagnostics = {
        'conservation_mse': conservation_mse,
        'max_conservation_error': max_conservation_error,
        'bankruptcy_occurred': bankruptcy_occurred,
        'first_bankruptcy_time': first_bankruptcy_time,
        'clamping_occurred': clamping_occurred
    }
    
    return diagnostics


# ============================================================================
# D — JACOBIAN & STABILITY ANALYSIS
# ============================================================================

def jacobian_and_stability_at(stocks_value, bonds_value, params, rf, rs, rb):
    """
    Compute Jacobian matrix and stability at a given point using actual rates.

    Inputs:
        stocks_value: S* (steady-state stocks)
        bonds_value:  B* (steady-state bonds)
        params:       parameter dict
        rf, rs, rb:   risk-free rate, stock return, bond return for that period

    For the linearized system with fixed rates:
        J11 = α_s*(r_s - r_f) - β_s - γ_s
        J12 = -β_s
        J21 = -β_b
        J22 = α_b*(r_b - r_f) - β_b - γ_b

    Returns:
        jacobian: 2x2 numpy array
        eigenvalues: array of eigenvalues
        stability_flag: True if all real parts < 0
    """
    # Extract parameters
    alpha_s = params.get("stock_sensitivity_to_excess_return", 0.5)
    alpha_b = params.get("bond_sensitivity_to_excess_return", 0.4)
    beta_s = params.get("cash_to_stocks_baseline_flow", 0.02)
    beta_b = params.get("cash_to_bonds_baseline_flow", 0.03)
    gamma_s = params.get("stock_natural_decay_rate", 0.01)
    gamma_b = params.get("bond_natural_decay_rate", 0.01)

    # Excess returns based on actual data
    stock_excess = rs - rf
    bond_excess = rb - rf

    # Jacobian entries
    J11 = alpha_s * stock_excess - beta_s - gamma_s
    J12 = -beta_s
    J21 = -beta_b
    J22 = alpha_b * bond_excess - beta_b - gamma_b

    jacobian = np.array([[J11, J12],
                         [J21, J22]])

    eigenvalues = eigvals(jacobian)
    stability_flag = all(np.real(eigenvalues) < 0)

    return jacobian, eigenvalues, stability_flag

# ============================================================================
# D2 — MONTHLY STEADY STATE (DATA-DRIVEN)
# ============================================================================

def steady_state_one_month(params, rf, rs, rb, initial_guess=None):
    """
    Solve dS/dt = 0, dB/dt = 0 for a single month using that month's rf, rs, rb.
    Returns (S, B, C) or None if the solver fails.
    """
    total_capital = params.get("total_capital", 1.0)

    alpha_s = params.get("stock_sensitivity_to_excess_return", 0.5)
    alpha_b = params.get("bond_sensitivity_to_excess_return", 0.4)
    beta_s = params.get("cash_to_stocks_baseline_flow", 0.02)
    beta_b = params.get("cash_to_bonds_baseline_flow", 0.03)
    gamma_s = params.get("stock_natural_decay_rate", 0.01)
    gamma_b = params.get("bond_natural_decay_rate", 0.01)

    def residuals(x):
        S, B = x
        C = total_capital - S - B
        stock_excess = rs - rf
        bond_excess  = rb - rf
        dS = alpha_s * stock_excess * S + beta_s * C - gamma_s * S
        dB = alpha_b * bond_excess  * B + beta_b * C - gamma_b * B
        return [dS, dB]

    if initial_guess is None:
        # simple neutral guess
        initial_guess = [0.5 * total_capital, 0.3 * total_capital]

    sol = root(residuals, x0=initial_guess, method="hybr")
    if not sol.success:
        return None

    S, B = sol.x
    C = total_capital - S - B
    return S, B, C


def steady_state_for_each_month(params, timeseries_df):
    """
    Compute steady state and stability for each month in the uploaded time-series.
    Returns a DataFrame with columns: date, S, B, C, eig1, eig2, stable.
    """
    results = []

    prev_guess = None

    for idx, row in timeseries_df.iterrows():
        rf = row["risk_free_rate"]
        rs = row["stock_return"]
        rb = row["bond_return"]

        ss = steady_state_one_month(params, rf, rs, rb, initial_guess=prev_guess)
        if ss is None:
            results.append({
                "date": idx,
                "S": np.nan,
                "B": np.nan,
                "C": np.nan,
                "eig1": np.nan,
                "eig2": np.nan,
                "stable": False
            })
            prev_guess = None
            continue

        S, B, C = ss
        prev_guess = [S, B]

        J, eigs, stable = jacobian_and_stability_at(S, B, params, rf, rs, rb)

        results.append({
            "date": idx,
            "S": S,
            "B": B,
            "C": C,
            "eig1": eigs[0],
            "eig2": eigs[1],
            "stable": stable
        })

    return pd.DataFrame(results)

# ============================================================================
# E — EMBEDDED UNIT TESTS
# ============================================================================

def run_unit_tests(params):
    """
    Run three embedded unit tests and return results.
    
    Test 1: Stagnation (α=β=γ=0)
    Test 2: Pure Decay (α=0, β=0, γ=0.1)
    Test 3: Pure Inflow (α=0, γ=0, β=0.1)
    
    Returns:
        dict with test results and messages
    """
    results = {}
    
    # Test 1: Stagnation
    try:
        test1_params = params.copy()
        test1_params.update({
            'stock_sensitivity_to_excess_return': 0.0,
            'bond_sensitivity_to_excess_return': 0.0,
            'cash_to_stocks_baseline_flow': 0.0,
            'cash_to_bonds_baseline_flow': 0.0,
            'stock_natural_decay_rate': 0.0,
            'bond_natural_decay_rate': 0.0,
            'total_capital': 1.0
        })
        
        initial_state = [0.5, 0.3]
        t_eval = np.linspace(0, 10, 101)
        
        sol = solve_ivp(
            fun=lambda t, y: capital_flow_ode(t, y, test1_params),
            t_span=(0, 10),
            y0=initial_state,
            t_eval=t_eval,
            method='RK45'
        )
        
        # Check that derivatives are essentially zero
        max_dS = max(abs(capital_flow_ode(t, [sol.y[0][i], sol.y[1][i]], test1_params)[0]) 
                    for i, t in enumerate(sol.t))
        max_dB = max(abs(capital_flow_ode(t, [sol.y[0][i], sol.y[1][i]], test1_params)[1]) 
                    for i, t in enumerate(sol.t))
        
        test1_pass = max(max_dS, max_dB) < 1e-12
        results['test1'] = {
            'pass': test1_pass,
            'message': f"Stagnation test: max|dS/dt|={max_dS:.2e}, max|dB/dt|={max_dB:.2e}"
        }
    except Exception as e:
        results['test1'] = {'pass': False, 'message': f"Test 1 failed: {str(e)}"}
    
    # Test 2: Pure Decay
    try:
        test2_params = params.copy()
        test2_params.update({
            'stock_sensitivity_to_excess_return': 0.0,
            'bond_sensitivity_to_excess_return': 0.0,
            'cash_to_stocks_baseline_flow': 0.0,
            'cash_to_bonds_baseline_flow': 0.0,
            'stock_natural_decay_rate': 0.1,
            'bond_natural_decay_rate': 0.0,
            'total_capital': 1.0
        })
        
        initial_state = [1.0, 0.0]
        
        sol = solve_ivp(
            fun=lambda t, y: capital_flow_ode(t, y, test2_params),
            t_span=(0, 1),
            y0=initial_state,
            t_eval=[1.0],
            method='RK45'
        )
        
        S_at_1 = sol.y[0][-1]
        expected = np.exp(-0.1)  # ≈ 0.904837
        error = abs(S_at_1 - expected)
        
        test2_pass = error < 1e-6
        results['test2'] = {
            'pass': test2_pass,
            'message': f"Pure decay test: S(1)={S_at_1:.6f}, expected={expected:.6f}, error={error:.2e}"
        }
    except Exception as e:
        results['test2'] = {'pass': False, 'message': f"Test 2 failed: {str(e)}"}
    
    # Test 3: Pure Inflow
    try:
        test3_params = params.copy()
        test3_params.update({
            'stock_sensitivity_to_excess_return': 0.0,
            'bond_sensitivity_to_excess_return': 0.0,
            'cash_to_stocks_baseline_flow': 0.1,
            'cash_to_bonds_baseline_flow': 0.0,
            'stock_natural_decay_rate': 0.0,
            'bond_natural_decay_rate': 0.0,
            'total_capital': 1.0
        })
        
        initial_state = [0.0, 0.0]
        
        sol = solve_ivp(
            fun=lambda t, y: capital_flow_ode(t, y, test3_params),
            t_span=(0, 2),
            y0=initial_state,
            t_eval=np.linspace(0, 2, 21),
            method='RK45'
        )
        
        S_increases = sol.y[0][-1] > sol.y[0][0]
        K = test3_params['total_capital']
        conservation_errors = [abs(sol.y[0][i] + sol.y[1][i] + 
                                  (K - sol.y[0][i] - sol.y[1][i]) - K) 
                              for i in range(len(sol.t))]
        max_conservation_error = max(conservation_errors)
        
        test3_pass = S_increases and max_conservation_error < 1e-8
        results['test3'] = {
            'pass': test3_pass,
            'message': f"Pure inflow test: S increased={S_increases}, conservation error={max_conservation_error:.2e}"
        }
    except Exception as e:
        results['test3'] = {'pass': False, 'message': f"Test 3 failed: {str(e)}"}
    
    return results


# ============================================================================
# F — QUARTERLY SUMMARY & INTERPRETATION
# ============================================================================

def generate_quarterly_summary(results_df, timeseries_df, params):
    """
    Generate quarter-by-quarter summary with interpretation sentences.
    Fixed to avoid deprecated timedelta units ('M', 'Y').
    Uses DateOffset for month jumps.
    """

    # ---------------------------
    # 1. Convert simulation time -> real calendar dates
    # ---------------------------
    start_date = pd.to_datetime(timeseries_df.index[0])
    base_date = start_date

    # Convert each time value t into: base_date + t months
    sim_dates = []
    for t in results_df["time"]:
        whole_months = int(t)
        fractional = t - whole_months

        # Add whole months
        d = base_date + pd.DateOffset(months=whole_months)

        # Add fractional month using approx 30 days (safe and acceptable)
        if fractional > 0:
            d = d + pd.Timedelta(days=30 * fractional)

        sim_dates.append(d)

    results_df["date"] = sim_dates

    # ---------------------------
    # 2. Assign calendar quarters
    # ---------------------------
    results_df["quarter"] = results_df["date"].dt.to_period("Q")

    quarterly_data = []

    # ---------------------------
    # 3. Group by quarter
    # ---------------------------
    for quarter, group in results_df.groupby("quarter"):
        quarter_start_date = group["date"].min()
        quarter_end_date = group["date"].max()

        S_start = group["stocks"].iloc[0]
        S_end = group["stocks"].iloc[-1]
        delta_S = S_end - S_start

        B_start = group["bonds"].iloc[0]
        B_end = group["bonds"].iloc[-1]
        delta_B = B_end - B_start

        avg_rf = group["risk_free_rate"].mean()
        avg_rs = group["stock_return"].mean()
        avg_rb = group["bond_return"].mean()

        total_flow_to_stocks = group["dS_dt"].sum()

        # Stability (at middle of quarter)
        mid = len(group) // 2
        mid_S  = group["stocks"].iloc[mid]
        mid_B  = group["bonds"].iloc[mid]
        mid_rf = group["risk_free_rate"].iloc[mid]
        mid_rs = group["stock_return"].iloc[mid]
        mid_rb = group["bond_return"].iloc[mid]

        try:
            _, eigenvalues, stable = jacobian_and_stability_at(
                mid_S, mid_B, params, mid_rf, mid_rs, mid_rb
            )
            stability = "Stable" if stable else "Unstable"
        except Exception:
            stability = "Unknown"


        quarterly_data.append({
            "quarter": str(quarter),
            "start_date": quarter_start_date.strftime("%Y-%m-%d"),
            "end_date": quarter_end_date.strftime("%Y-%m-%d"),
            "delta_S": delta_S,
            "delta_B": delta_B,
            "avg_rf": avg_rf,
            "avg_rs": avg_rs,
            "avg_rb": avg_rb,
            "total_flow_to_stocks": total_flow_to_stocks,
            "stability": stability
        })

    quarterly_df = pd.DataFrame(quarterly_data)

    # ---------------------------
    # 4. Build one-line interpretations
    # ---------------------------
    interpretations = []

    for _, row in quarterly_df.iterrows():
        quarter_label = row["quarter"]
        year = quarter_label[:4]
        q_number = quarter_label[-2:]

        rotation = "rotation into stocks" if row["delta_S"] > 0 else "flight to safety"
        avg_spread = row["avg_rs"] - row["avg_rf"]

        sentence = (
            f"{q_number} {year}: Net {rotation} — "
            f"ΔS = {row['delta_S']:.3%}; "
            f"driven by avg spread (r_s − r_f) = {avg_spread:.3%}. "
            f"System stability: {row['stability']}."
        )

        interpretations.append(sentence)

    return quarterly_df, interpretations



# ============================================================================
# STEADY STATE & JACOBIAN (ORIGINAL FUNCTIONS)
# ============================================================================

def steady_state(params, fixed_risk_free_rate=None, initial_guess=None):
    """Find steady-state capital allocation."""
    if fixed_risk_free_rate is None:
        fixed_risk_free_rate = params.get("risk_free_rate_baseline", 0.04)
    
    local_params = dict(params)
    local_params["risk_free_rate_type"] = "constant"
    local_params["risk_free_rate_baseline"] = fixed_risk_free_rate
    
    def residuals(variables):
        stocks_allocation, bonds_allocation = variables
        total_capital = local_params.get("total_capital", 1.0)
        cash_allocation = total_capital - stocks_allocation - bonds_allocation
        
        if cash_allocation < 0:
            return [1e3 * (stocks_allocation + bonds_allocation - total_capital), 
                    1e3 * (stocks_allocation + bonds_allocation - total_capital)]
        
        current_stock_return = stock_return(fixed_risk_free_rate, local_params)
        current_bond_return = bond_return(fixed_risk_free_rate, local_params)
        
        alpha_s = local_params.get("stock_sensitivity_to_excess_return", 0.5)
        alpha_b = local_params.get("bond_sensitivity_to_excess_return", 0.4)
        beta_s = local_params.get("cash_to_stocks_baseline_flow", 0.02)
        beta_b = local_params.get("cash_to_bonds_baseline_flow", 0.03)
        gamma_s = local_params.get("stock_natural_decay_rate", 0.01)
        gamma_b = local_params.get("bond_natural_decay_rate", 0.01)
        
        stock_excess_return = current_stock_return - fixed_risk_free_rate
        bond_excess_return = current_bond_return - fixed_risk_free_rate
        
        stocks_derivative = (alpha_s * stock_excess_return * stocks_allocation + 
                            beta_s * cash_allocation - 
                            gamma_s * stocks_allocation)
        
        bonds_derivative = (alpha_b * bond_excess_return * bonds_allocation + 
                           beta_b * cash_allocation - 
                           gamma_b * bonds_allocation)
        
        return [stocks_derivative, bonds_derivative]
    
    if initial_guess is None:
        current_stock_return = stock_return(fixed_risk_free_rate, params)
        current_bond_return = bond_return(fixed_risk_free_rate, params)
        total_return = max(current_stock_return + current_bond_return, 1e-6)
        total_capital = params.get("total_capital", 1.0)
        initial_guess = [total_capital * current_stock_return / total_return, 
                        total_capital * current_bond_return / total_return]
    
    solution = root(residuals, x0=initial_guess, method="hybr")
    if not solution.success:
        raise RuntimeError(f"Steady-state solver failed: {solution.message}")
    
    stocks_steady, bonds_steady = solution.x
    total_capital = params.get("total_capital", 1.0)
    cash_steady = total_capital - stocks_steady - bonds_steady
    
    return np.array([stocks_steady, bonds_steady, cash_steady])


def run_simulation(params, initial_state=None, time_points=None):
    """Run the capital flow simulation (original continuous integration fallback)."""
    time_horizon = params.get("simulation_time_horizon", 200.0)
    if time_points is None:
        time_points = np.linspace(0.0, time_horizon, 1001)
    
    if initial_state is None:
        total_capital = params.get("total_capital", 1.0)
        initial_stocks = params.get("initial_stocks_fraction", 0.5)
        initial_bonds = params.get("initial_bonds_fraction", 0.3)
        initial_state = [initial_stocks * total_capital, initial_bonds * total_capital]
    
    solution = solve_ivp(
        fun=lambda t, y: capital_flow_ode(t, y, params),
        t_span=(time_points[0], time_points[-1]),
        y0=initial_state,
        t_eval=time_points,
        method="RK45",
        atol=1e-8, 
        rtol=1e-6
    )
    
    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")
    
    stocks_over_time = solution.y[0]
    bonds_over_time = solution.y[1]
    total_capital = params.get("total_capital", 1.0)
    cash_over_time = total_capital - stocks_over_time - bonds_over_time
    
    results_dataframe = pd.DataFrame({
        "time": solution.t,
        "stocks": stocks_over_time,
        "bonds": bonds_over_time,
        "cash": cash_over_time,
        "risk_free_rate": [risk_free_rate(t, params) for t in solution.t],
        "stock_return": [stock_return(risk_free_rate(t, params), params) for t in solution.t],
        "bond_return": [bond_return(risk_free_rate(t, params), params) for t in solution.t],
    })
    
    return solution, results_dataframe


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(page_title="Capital Flow Model", layout="wide", page_icon="📈")
    
    st.title("📈 Enhanced Capital Flow Model")
    st.markdown("*Monthly time-series integration with diagnostics, stability analysis, and quarterly reporting*")
    
    # ========================================================================
    # SIDEBAR: Configuration
    # ========================================================================
    
    st.sidebar.header("⚙️ Configuration")
    
    # Time-series upload section
    st.sidebar.subheader("📊 Time-Series Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Monthly CSV (date, rf, rs, rb)",
        type=['csv'],
        help="Upload a CSV with columns: date, rf (risk-free), rs (stock return), rb (bond return)"
    )
    
    timeseries_df = None
    if uploaded_file is not None:
        is_percent = st.sidebar.checkbox(
            "Rates in percent?",
            value=True,
            help="Check if rates are in percent (will divide by 100)"
        )
        is_annual = st.sidebar.checkbox(
            "Rates are annual?",
            value=True,
            help="Check if rates are annual (will divide by 12 for monthly)"
        )
        
        try:
            raw_df = pd.read_csv(uploaded_file)
            timeseries_df = preprocess_timeseries(raw_df, is_percent, is_annual)
            st.sidebar.success(f"✅ Loaded {len(timeseries_df)} months of data")
            st.sidebar.info(f"Date range: {timeseries_df.index[0].strftime('%Y-%m')} to {timeseries_df.index[-1].strftime('%Y-%m')}")
        except Exception as e:
            st.sidebar.error(f"❌ Error processing CSV: {str(e)}")
            timeseries_df = None
    
    # Integration settings
    if timeseries_df is not None:
        st.sidebar.subheader("🔧 Integration Settings")
        steps_per_month = st.sidebar.slider(
            "Steps per month",
            min_value=5,
            max_value=50,
            value=10,
            help="Number of evaluation points per month"
        )
    
    st.sidebar.markdown("---")
    
    # Parameters
    st.sidebar.header("📋 Model Parameters")
    params = dict(DEFAULT_PARAMS)
    
    with st.sidebar.expander("🎯 Core Parameters", expanded=True):
        params["stock_sensitivity_to_excess_return"] = st.number_input(
            "Stock Sensitivity α_s",
            min_value=0.0, max_value=2.0,
            value=params["stock_sensitivity_to_excess_return"],
            step=0.1, format="%.2f"
        )
        
        params["bond_sensitivity_to_excess_return"] = st.number_input(
            "Bond Sensitivity α_b",
            min_value=0.0, max_value=2.0,
            value=params["bond_sensitivity_to_excess_return"],
            step=0.1, format="%.2f"
        )
        
        params["cash_to_stocks_baseline_flow"] = st.number_input(
            "Cash→Stocks Flow β_s",
            min_value=0.0, max_value=1.0,
            value=params["cash_to_stocks_baseline_flow"],
            step=0.01, format="%.4f"
        )
        
        params["cash_to_bonds_baseline_flow"] = st.number_input(
            "Cash→Bonds Flow β_b",
            min_value=0.0, max_value=1.0,
            value=params["cash_to_bonds_baseline_flow"],
            step=0.01, format="%.4f"
        )
        
        params["stock_natural_decay_rate"] = st.number_input(
            "Stock Decay γ_s",
            min_value=0.0, max_value=1.0,
            value=params["stock_natural_decay_rate"],
            step=0.01, format="%.4f"
        )
        
        params["bond_natural_decay_rate"] = st.number_input(
            "Bond Decay γ_b",
            min_value=0.0, max_value=1.0,
            value=params["bond_natural_decay_rate"],
            step=0.01, format="%.4f"
        )
    
    with st.sidebar.expander("⚙️ Initial Conditions"):
        params["initial_stocks_fraction"] = st.number_input(
            "Initial Stocks (fraction)",
            min_value=0.0, max_value=1.0,
            value=params["initial_stocks_fraction"],
            step=0.1, format="%.2f"
        )
        
        params["initial_bonds_fraction"] = st.number_input(
            "Initial Bonds (fraction)",
            min_value=0.0, max_value=1.0,
            value=params["initial_bonds_fraction"],
            step=0.1, format="%.2f"
        )
        
        params["total_capital"] = st.number_input(
            "Total Capital K",
            min_value=0.1, max_value=10.0,
            value=params["total_capital"],
            step=0.1, format="%.2f"
        )
    with st.sidebar.expander("📐 Steady-State / Return Parameters", expanded=False):
        params["baseline_stock_return"] = st.number_input(
            "Baseline Stock Return r_s⁰",
            min_value=-1.0, max_value=1.0,
            value=params["baseline_stock_return"],
            step=0.01, format="%.4f",
            help="Long-run annual stock return used in steady-state when no data is given."
        )

        params["baseline_bond_return"] = st.number_input(
            "Baseline Bond Return r_b⁰",
            min_value=-1.0, max_value=1.0,
            value=params["baseline_bond_return"],
            step=0.01, format="%.4f",
            help="Long-run annual bond return used in steady-state when no data is given."
        )

        params["lambda_s"] = st.number_input(
            "Stock Rate Sensitivity λ_s",
            min_value=-5.0, max_value=5.0,
            value=params.get("lambda_s", -0.10),
            step=0.01, format="%.3f",
            help="How much the expected stock return moves when the risk-free rate moves."
        )

        params["risk_free_rate_baseline"] = st.number_input(
            "Baseline Risk-Free Rate r_f⁰",
            min_value=-0.5, max_value=1.0,
            value=params["risk_free_rate_baseline"],
            step=0.01, format="%.4f",
            help="Baseline risk-free rate used in steady-state and default simulations."
        )

        params["risk_free_rate_shock_magnitude"] = st.number_input(
            "Shock Magnitude Δr_f",
            min_value=-0.5, max_value=1.0,
            value=params["risk_free_rate_shock_magnitude"],
            step=0.01, format="%.4f",
            help="Size of permanent rate shock used in the pre/post steady-state comparison."
        )

    
    # Only show these if no timeseries
    if timeseries_df is None:
        with st.sidebar.expander("📉 Risk-Free Rate (No Timeseries)"):
            params["risk_free_rate_baseline"] = st.number_input(
                "Baseline Rate",
                min_value=0.0, max_value=1.0,
                value=params["risk_free_rate_baseline"],
                step=0.01, format="%.4f"
            )
            
            params["risk_free_rate_type"] = st.selectbox(
                "Rate Type",
                ["constant", "step", "sin"],
                index=["constant", "step", "sin"].index(params["risk_free_rate_type"])
            )
            
            if params["risk_free_rate_type"] != "constant":
                params["risk_free_rate_shock_magnitude"] = st.number_input(
                    "Shock Magnitude",
                    min_value=0.0, max_value=1.0,
                    value=params["risk_free_rate_shock_magnitude"],
                    step=0.01, format="%.4f"
                )
                
                if params["risk_free_rate_type"] == "step":
                    params["risk_free_rate_shock_time"] = st.number_input(
                        "Shock Time",
                        min_value=0.0, max_value=500.0,
                        value=params["risk_free_rate_shock_time"],
                        step=1.0, format="%.1f"
                    )
            
            params["simulation_time_horizon"] = st.number_input(
                "Time Horizon",
                min_value=1.0, max_value=1000.0,
                value=params["simulation_time_horizon"],
                step=10.0, format="%.1f"
            )
    
    # ========================================================================
    # MAIN AREA: Action Buttons
    # ========================================================================
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    
    with col1:
        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Running simulation..."):
                try:
                    if timeseries_df is not None:
                        # Piecewise monthly integration
                        results_df = run_simulation_with_timeseries(
                            params, timeseries_df, steps_per_month
                        )
                        diagnostics = add_diagnostics(results_df, params)
                        
                        st.session_state['results_df'] = results_df
                        st.session_state['diagnostics'] = diagnostics
                        st.session_state['timeseries_df'] = timeseries_df
                        st.session_state['params'] = params
                        st.session_state['using_timeseries'] = True
                    else:
                        # Continuous integration fallback
                        solution, results_df = run_simulation(params)
                        st.session_state['results_df'] = results_df
                        st.session_state['params'] = params
                        st.session_state['using_timeseries'] = False
                    
                    st.success("✅ Simulation completed!")
                except Exception as e:
                    st.error(f"❌ Simulation failed: {str(e)}")
    
    with col2:
        if st.button("📊 Calculate Steady State", use_container_width=True):
            with st.spinner("Calculating..."):
                try:
                    pre_shock_rate = params.get("risk_free_rate_baseline", 0.04)
                    post_shock_rate = pre_shock_rate + params.get("risk_free_rate_shock_magnitude", 0.02)
                    
                    pre_shock_ss = steady_state(params, fixed_risk_free_rate=pre_shock_rate)
                    post_shock_ss = steady_state(params, fixed_risk_free_rate=post_shock_rate, 
                                                 initial_guess=pre_shock_ss[:2])

                    # Compute rs, rb implied by the post-shock rf for this theoretical analysis
                    post_rs = stock_return(post_shock_rate, params)
                    post_rb = bond_return(post_shock_rate, params)
                    
                    jacobian_matrix, eigenvalues, stability_flag = jacobian_and_stability_at(
                        post_shock_ss[0], post_shock_ss[1], params,
                        post_shock_rate, post_rs, post_rb
                    )

                    
                    st.session_state['steady_state_results'] = {
                        'pre_shock_rate': pre_shock_rate,
                        'post_shock_rate': post_shock_rate,
                        'pre_shock_ss': pre_shock_ss,
                        'post_shock_ss': post_shock_ss,
                        'eigenvalues': eigenvalues,
                        'stability_flag': stability_flag,
                        'jacobian': jacobian_matrix
                    }
                    st.success("✅ Steady state calculated!")
                except Exception as e:
                    st.error(f"❌ Calculation failed: {str(e)}")
    
    with col3:
        if st.button("🧪 Run Unit Tests", use_container_width=True):
            with st.spinner("Running tests..."):
                try:
                    test_results = run_unit_tests(params)
                    st.session_state['test_results'] = test_results
                    
                    all_passed = all(result['pass'] for result in test_results.values())
                    if all_passed:
                        st.success("✅ All tests passed!")
                    else:
                        st.warning("⚠️ Some tests failed")
                except Exception as e:
                    st.error(f"❌ Tests failed: {str(e)}")
    with col4:
        if st.button("📅 Monthly Steady States", use_container_width=True):
            with st.spinner("Computing monthly steady states..."):
                try:
                    if timeseries_df is None:
                        st.error("Upload time-series data first.")
                    else:
                        ss_monthly = steady_state_for_each_month(params, timeseries_df)
                        st.session_state["monthly_ss"] = ss_monthly
                        st.success("✅ Monthly steady states computed!")
                except Exception as e:
                    st.error(f"❌ Monthly steady-state computation failed: {str(e)}")

    # ========================================================================
    # DISPLAY: Test Results
    # ========================================================================
    
    if 'test_results' in st.session_state:
        st.markdown("---")
        st.subheader("🧪 Unit Test Results")
        
        test_results = st.session_state['test_results']
        
        cols = st.columns(3)
        for idx, (test_name, result) in enumerate(test_results.items()):
            with cols[idx]:
                if result['pass']:
                    st.success(f"✅ {test_name.upper()}")
                else:
                    st.error(f"❌ {test_name.upper()}")
                st.caption(result['message'])
    
    # ========================================================================
    # DISPLAY: Steady State
    # ========================================================================
    
    if 'steady_state_results' in st.session_state:
        st.markdown("---")
        st.subheader("📐 Steady State Analysis")
        
        ss_results = st.session_state['steady_state_results']
        
        col_ss1, col_ss2 = st.columns(2)
        
        with col_ss1:
            st.markdown("##### Pre-Shock Steady State")
            st.metric("Risk-Free Rate", f"{ss_results['pre_shock_rate']:.4f}")
            st.metric("Stocks", f"{ss_results['pre_shock_ss'][0]:.6f}")
            st.metric("Bonds", f"{ss_results['pre_shock_ss'][1]:.6f}")
            st.metric("Cash", f"{ss_results['pre_shock_ss'][2]:.6f}")
        
        with col_ss2:
            st.markdown("##### Post-Shock Steady State")
            st.metric("Risk-Free Rate", f"{ss_results['post_shock_rate']:.4f}")
            st.metric("Stocks", f"{ss_results['post_shock_ss'][0]:.6f}")
            st.metric("Bonds", f"{ss_results['post_shock_ss'][1]:.6f}")
            st.metric("Cash", f"{ss_results['post_shock_ss'][2]:.6f}")
        
        st.markdown("##### Stability Analysis")
        eigs = ss_results['eigenvalues']
        stability_status = "Stable ✅" if ss_results['stability_flag'] else "Unstable ⚠️"
        
        st.write(f"**Eigenvalues:** λ₁ = {eigs[0]:.6f}, λ₂ = {eigs[1]:.6f}")
        st.write(f"**Real parts:** Re(λ₁) = {np.real(eigs[0]):.6f}, Re(λ₂) = {np.real(eigs[1]):.6f}")
        st.write(f"**System Stability:** {stability_status}")
    
    # ========================================================================
    # DISPLAY: Simulation Results
    # ========================================================================
    
    if 'results_df' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Simulation Results")
        
        results_df = st.session_state['results_df']
        using_timeseries = st.session_state.get('using_timeseries', False)
        
        # Show diagnostics if available
        if 'diagnostics' in st.session_state:
            diagnostics = st.session_state['diagnostics']
            
            st.markdown("##### 🔍 Diagnostics")
            
            diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)
            
            with diag_col1:
                st.metric("Conservation MSE", f"{diagnostics['conservation_mse']:.2e}")
            
            with diag_col2:
                st.metric("Max Conservation Error", f"{diagnostics['max_conservation_error']:.2e}")
                if diagnostics['max_conservation_error'] > 1e-5:
                    st.warning("⚠️ Large conservation error detected!")
            
            with diag_col3:
                if diagnostics['bankruptcy_occurred']:
                    st.error("❌ Bankruptcy Event")
                    if diagnostics['first_bankruptcy_time'] is not None:
                        st.caption(f"First at t={diagnostics['first_bankruptcy_time']:.2f}")
                else:
                    st.success("✅ No Bankruptcy")
            
            with diag_col4:
                if diagnostics['clamping_occurred']:
                    st.info("ℹ️ Tiny negatives clamped")
                else:
                    st.success("✅ No clamping needed")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Capital Allocation", 
            "💹 Rates & Returns", 
            "🔄 Phase Diagram",
            "📅 Quarterly Summary",
            "📋 Data Table",
            "🧮 Monthly Steady State"
        ])

        
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(results_df["time"], results_df["stocks"], label="Stocks", linewidth=2)
            ax1.plot(results_df["time"], results_df["bonds"], label="Bonds", linewidth=2)
            ax1.plot(results_df["time"], results_df["cash"], label="Cash", linewidth=2)
            ax1.set_xlabel("Time (months)" if using_timeseries else "Time", fontsize=12)
            ax1.set_ylabel("Capital Allocation", fontsize=12)
            ax1.set_title("Capital Allocation Over Time", fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            fig1.tight_layout()
            st.pyplot(fig1)
        
        with tab2:
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(results_df["time"], results_df["risk_free_rate"], 
                    label="Risk-Free Rate", linewidth=2)
            ax2.plot(results_df["time"], results_df["bond_return"], 
                    label="Bond Return", linewidth=2)
            ax2.plot(results_df["time"], results_df["stock_return"], 
                    label="Stock Return", linewidth=2)
            ax2.set_xlabel("Time (months)" if using_timeseries else "Time", fontsize=12)
            ax2.set_ylabel("Monthly Return" if using_timeseries else "Annual Return", fontsize=12)
            ax2.set_title("Interest Rates and Asset Returns", fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            st.pyplot(fig2)
        
        with tab3:
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            ax3.plot(results_df["stocks"], results_df["bonds"], linewidth=2, color='purple')
            ax3.scatter(results_df["stocks"].iloc[0], results_df["bonds"].iloc[0], 
                       color='green', s=100, label='Start', zorder=5)
            ax3.scatter(results_df["stocks"].iloc[-1], results_df["bonds"].iloc[-1], 
                       color='red', s=100, label='End', zorder=5)
            ax3.set_xlabel("Stocks", fontsize=12)
            ax3.set_ylabel("Bonds", fontsize=12)
            ax3.set_title("Phase Diagram: Stocks vs Bonds", fontsize=14, fontweight='bold')
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)
            fig3.tight_layout()
            st.pyplot(fig3)
        
        with tab4:
            if using_timeseries and 'timeseries_df' in st.session_state:
                try:
                    quarterly_df, interpretations = generate_quarterly_summary(
                        results_df.copy(), 
                        st.session_state['timeseries_df'],
                        st.session_state['params']
                    )
                    
                    st.markdown("##### 📅 Quarterly Aggregations")
                    st.dataframe(quarterly_df, use_container_width=True)
                    
                    st.markdown("##### 💬 Interpretations")
                    for interpretation in interpretations:
                        st.info(interpretation)
                    
                except Exception as e:
                    st.error(f"Could not generate quarterly summary: {str(e)}")
            else:
                st.info("Quarterly summary available only for time-series simulations")
        
        with tab5:
            st.dataframe(results_df, use_container_width=True)
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="capital_flow_results_enhanced.csv",
                mime="text/csv",
            )
    

        with tab6:
            if "monthly_ss" in st.session_state:
                st.markdown("##### 🧮 Monthly Steady-State (Data-Driven)")
                st.dataframe(st.session_state["monthly_ss"], use_container_width=True)
            else:
                st.info("Click '📅 Monthly Steady States' above to compute monthly steady-state and stability.")
    else:
        st.info("👆 Click 'Run Simulation' to see results")



if __name__ == "__main__":
    main()
