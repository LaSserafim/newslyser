# NEWSLYSER // CPI QUANTITATIVE ENGINE & EVENT HORIZON TERMINAL

An institutional econometric framework, transmission model, and real-time event study terminal engineered for U.S. Consumer Price Index (CPI) releases and their cross-asset transmission into Gold ($XAUUSD$), the U.S. Dollar Index ($DXY$), and U.S. Treasury yields.

---

## 1. Executive Summary & Failure of Naive Models

A popular retail approach to CPI trading regresses Gold returns directly against raw Headline CPI surprises:
$$r_{\text{XAU}} = \alpha + \beta (A - F) + \epsilon$$

When backtested out-of-sample on authentic macroeconomic releases, **this naive linear model yields an out-of-sample directional accuracy of only 49.19% ($\text{AUC} = 0.482$)**—indistinguishable from a random coin flip.

### Structural Causes of Failure:
1. **Consensus Rounding Distortion:** Calendar forecasts round to $0.1\%$ increments. A print of $0.24\%$ is reported as $0.2\%$ (dovish shock), while $0.26\%$ is reported as $0.3\%$ (hawkish shock), creating discrete survey noise.
2. **Aggregation Fallacy:** The Federal Reserve's policy reaction function does not trade Headline CPI; it filters out volatile components and focuses on **Core CPI** and **Supercore (Services ex-Energy & Shelter)**.
3. **Broken Transmission Linearity:** Gold does not trade CPI in a vacuum; it prices **10-Year Real Yields ($10Y\text{ TIPS}$) and the U.S. Dollar Index ($DXY$)**.
4. **Non-Linear Threshold Dynamics:** Small surprises ($|Z| < 1.0\sigma$) are noise dominated by liquidity provider bid-ask bounce and whipsaws. Only tail shocks ($|Z| \ge 1.5\sigma$) produce statistically persistent Post-Announcement Drift (PEAD).

---

## 2. Refined Econometric Architecture

### A. Multi-Component Standardized Surprise
Instead of arithmetic difference, compute standardized $Z$-scores across components using expanding historical windows without lookahead bias:
$$Z_{\text{comp}} = 0.60 Z_{\text{core}} + 0.40 Z_{\text{headline}}$$
Where:
$$Z = \frac{\text{Actual} - \text{Consensus}}{\sigma_{\text{survey-dispersion}}}$$

### B. Two-Stage Transmission Formulation
* **Stage 1 (Macro Shock to Real Yields & USD):**
  $$\Delta y_{\text{real}, 10Y} = \alpha_1 + \beta_1 Z_{\text{comp}} + \epsilon_1$$
  $$\Delta \text{DXY} = \alpha_2 + \beta_2 Z_{\text{comp}} + \epsilon_2$$
* **Stage 2 (Asset Pricing on Gold):**
  $$r_{\text{XAU}} = \alpha_3 + \gamma_1 \Delta y_{\text{real}, 10Y} + \gamma_2 \Delta \text{DXY} + \epsilon_3$$

Empirical parameters calibrated with White HC3 heteroskedasticity-consistent covariance ($p < 0.0001, R^2 = 0.2139$, correlation with Gold $r = -0.463$, correlation with DXY $r = -0.514$).

### C. Regime Classification
* **Noise Zone ($|Z_{\text{comp}}| < 1.0\sigma$):** High mean-reversion probability; strictly fade knee-jerk liquidity spikes back to the $T_0$ anchor price.
* **Momentum Shock Zone ($|Z_{\text{comp}}| \ge 1.0\sigma$):** Directional consistency reaches **70.0%** when verified by 2Y Yield and DXY confirmation.

---

## 3. The 3-Wave Execution Protocol

```
T0 (08:30 ET)                 T+1m            T+2m to T+15m           T+15m to T+60m
──────┼────────────────────────┼────────────────────┼───────────────────────►
      │ Wave 1: Algorithmic    │ Wave 2: Digestion  │ Wave 3: Institutional Drift
      │ Machine wire feeds     │ Core vs Supercore  │ PEAD Trend Continuation
      │ Spreads: $2.00–$5.00   │ Spreads compress   │ Normal spreads
      │ NO MARKET ORDERS       │ Filter fakeouts    │ Enter on VWAP pullbacks
```

| Wave | Time Window | Market Microstructure | Operational Directive |
| :--- | :--- | :--- | :--- |
| **Wave 1** | $T_0$ to $T+60\text{s}$ | Machine-readable feeds trigger market orders. Spreads widen 10x–25x ($2.00–$5.00 on Gold). | **NO MARKET ORDERS.** Slippage creates negative expectancy. |
| **Wave 2** | $T+2\text{m}$ to $T+15\text{m}$ | Desks digest component details. False headline moves get faded. | **VALIDATION WINDOW.** Check 2Y Yield & DXY. |
| **Wave 3** | $T+15\text{m}$ to $T+60\text{m}$ | Macro portfolio rebalancing and Post-Announcement Drift (PEAD). | **EXECUTION WINDOW.** Enter on retests/VWAP pullbacks. |

---

## 4. Authentic Verified Data Sources
* **U.S. Bureau of Labor Statistics (BLS):** Official release schedule and monthly index series (`CPIAUCSL`, `CPILFESL`, `CUSR0000SASLE`, `CUSR0000SEHC`).
* **Federal Reserve Bank of St. Louis (FRED):** 10-Year TIPS Real Yield (`DFII10`), 2-Year Treasury Yield (`DGS2`), Broad Dollar Index (`DTWEXBGS`).
* **Yahoo Finance:** High-precision daily OHLCV on Gold Futures (`GC=F`), U.S. Dollar Index (`DX-Y.NYB`), 2-Year Treasury Note Futures (`ZT=F`), and 10-Year Yield (`^TNX`).
* **ForexFactory / Investing.com:** Consensus forecast surveys and Cleveland Fed Nowcasts.

---

## 5. Quickstart & Localhost Access

### Requirements
* Python 3.9+
* Dependencies: `pip install -r requirements.txt`

### Launch Localhost Server
```bash
./run.sh
# Server starts immediately on http://127.0.0.1:8888
```

Open your browser at `http://127.0.0.1:8888` to access the interactive tactical telemetry console.
