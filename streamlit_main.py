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

# ============================================================================
# B — PIECEWISE MONTHLY INTEGRATION
# ============================================================================

# ============================================================================
# C — CONSERVATION DIAGNOSTICS & NEGATIVE HANDLING
# ============================================================================

# ============================================================================
# D — JACOBIAN & STABILITY ANALYSIS
# ============================================================================

# ============================================================================
# D2 — MONTHLY STEADY STATE (DATA-DRIVEN)
# ============================================================================

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
        
        initial_state = [0.0, 0.0, 1.0]
        
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
                                  sol.y[2][i] - K)
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

# ============================================================================
# STEADY STATE & JACOBIAN (ORIGINAL FUNCTIONS)
# ============================================================================

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(page_title="Capital Flow Model", layout="wide", page_icon="📈")
    
    st.title("📈 Enhanced Capital Flow Model")
    st.markdown("*Funded allocation transfers — time in years, fixed capital budget K*")
    st.info("Returns guide allocation transfers; this model does not compound wealth or price bonds by duration. Uploaded monthly values describe historical forcing, not a trading backtest.")
    
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
            help="Annual inputs stay annual; monthly continuous rates are multiplied by 12."
        )
        
        rate_convention = st.sidebar.selectbox("Input convention", ["continuous", "effective"],
            help="Effective returns are converted with log1p. Use bond total-return assumptions, not bond yields.")

        try:
            raw_df = pd.read_csv(uploaded_file)
            timeseries_df = preprocess_timeseries(raw_df, is_percent, is_annual, rate_convention)
            st.sidebar.success(f"✅ Loaded {len(timeseries_df)} months of data")
            st.sidebar.caption(f"Past-only carried rate cells: {timeseries_df.attrs.get('imputed_cells', 0)}")
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
            "Cash→Stocks Flow β_s (per year)",
            min_value=0.0, max_value=1.0,
            value=params["cash_to_stocks_baseline_flow"],
            step=0.01, format="%.4f"
        )
        
        params["cash_to_bonds_baseline_flow"] = st.number_input(
            "Cash→Bonds Flow β_b (per year)",
            min_value=0.0, max_value=1.0,
            value=params["cash_to_bonds_baseline_flow"],
            step=0.01, format="%.4f"
        )
        
        params["stock_natural_decay_rate"] = st.number_input(
            "Stock Decay γ_s (per year)",
            min_value=0.0, max_value=1.0,
            value=params["stock_natural_decay_rate"],
            step=0.01, format="%.4f"
        )
        
        params["bond_natural_decay_rate"] = st.number_input(
            "Bond Decay γ_b (per year)",
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
                        "Shock Time (years)",
                        min_value=0.0, max_value=500.0,
                        value=params["risk_free_rate_shock_time"],
                        step=1.0, format="%.1f"
                    )
            
            params["simulation_time_horizon"] = st.number_input(
                "Time Horizon (years)",
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
                        st.session_state.pop('monthly_ss', None)
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
                        st.session_state['diagnostics'] = add_diagnostics(results_df, params)
                        st.session_state.pop('timeseries_df', None)
                        st.session_state.pop('monthly_ss', None)
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
        stability_status = "Frozen-rate asymptotically stable ✅" if ss_results['stability_flag'] else "Not strictly asymptotically stable"
        
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
                    st.error("❌ Negative allocation detected")
                    if diagnostics['first_bankruptcy_time'] is not None:
                        st.caption(f"First at t={diagnostics['first_bankruptcy_time']:.2f}")
                else:
                    st.success("✅ Allocations nonnegative")
            
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
            ax1.set_xlabel("Time (years)", fontsize=12)
            ax1.set_ylabel("Capital Allocation", fontsize=12)
            ax1.set_title("Capital Allocation Over Time", fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            fig1.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
        
        with tab2:
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            rate_plot = ax2.plot if not using_timeseries and st.session_state["params"].get("risk_free_rate_type") == "sin" else ax2.step
            rate_options = {} if rate_plot == ax2.plot else {"where": "pre" if using_timeseries else "post"}
            rate_plot(results_df["time"], results_df["risk_free_rate"],
                    label="Risk-Free Rate", linewidth=2, **rate_options)
            rate_plot(results_df["time"], results_df["bond_return"],
                    label="Bond Return", linewidth=2, **rate_options)
            rate_plot(results_df["time"], results_df["stock_return"],
                    label="Stock Return", linewidth=2, **rate_options)
            ax2.set_xlabel("Time (years)", fontsize=12)
            ax2.set_ylabel("Annual continuous rate", fontsize=12)
            ax2.set_title("Interest Rates and Asset Returns", fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        
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
            plt.close(fig3)
        
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
