# Funded capital-allocation model (v2)

A fixed-budget allocation experiment for stocks, bonds and cash. The corrected
core is capital_model.py; streamlit_main.py imports it for simulation, equilibrium
and quarterly reporting. This is a model revision: positive excess-return
incentives now fund stock/bond entry from cash instead of creating unfunded
allocation growth. Returns do not compound wealth, and bond duration pricing is
not modeled.

For each asset, e=alpha*(asset_return-risk_free_rate), entry rate
u=beta+max(e,0), and exit rate d=gamma+max(-e,0). Stocks evolve as
S'=u_s*C-d_s*S, bonds analogously, and cash receives every exit and funds every
entry. The three balances evolve independently and sum to K. Nonnegative
coefficients make every boundary point inward; output is not silently clamped.

All model time is in years; beta and gamma are per-year rates. Monthly intervals
span 1/12 year. Uploaded annual continuous rates stay annual, monthly continuous
rates multiply by 12. Effective returns are converted using log1p before
annualization. A bond yield must not be substituted for a bond total-return
incentive. Monthly observations label historical forcing, not a causal strategy.
Missing cells can carry prior values for at most two months, with the carried
cell count displayed; longer or leading gaps fail. The last whole row of each
month is retained, rather than mixing columns from different observations.

Fixed-rate intervals use a matrix exponential, including explicit separation at
step shocks. Smooth sinusoidal forcing uses a tightly controlled adaptive
solver. Equilibria solve the same funded equations and budget constraint;
nonunique equilibria are reported. Frozen-rate eigenvalues describe that fixed
system, not general time-varying stability.

Quarterly net stock flow equals the sum of interval balance increments, hence
S(end)-S(start). It is not the gross sum of purchases and does not depend on the
number of derivative samples. Dates use real calendar-month boundaries; shared
simulation boundaries appear once. At boundaries, monthly rate/derivative
columns refer to the interval ending at that row.

```bash
python -m pip install -r requirements.txt pytest
python -m pytest tests -q
python -m streamlit run streamlit_main.py
```

The app's default horizon is now 20 years with a shock at year 2. These are
illustrative parameters, not estimates fitted to real asset flows. Older output
or equations are not interchangeable with model v2.

Original project contributors:

Abhishek Rai
Alok Silswal
Ashutosh
Divyansh Singodia
Om Ujjwal Singh
Ritin Sundriyal
Shreyansh Chauhan
