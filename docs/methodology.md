# Methodology and full results

This document expands on the [README](../README.md) with every number behind our decisions. All of them are produced by the scripts in [`scripts/`](../scripts) and saved in [`results/`](../results).

## Contents

1. [Data](#1-data)
2. [Round 1: direction in a calm market](#2-round-1-direction-in-a-calm-market)
3. [Round 2: the flash crash](#3-round-2-the-flash-crash)
4. [Round 3: the trading book](#4-round-3-the-trading-book)
5. [Post-event analysis](#5-post-event-analysis)
6. [Metrics glossary](#6-metrics-glossary)

## 1. Data

One file per round, one row per trading day (900 days, July 2016 to December 2019).

| Column | Meaning |
| --- | --- |
| `open`, `high`, `low`, `close`, `volume` | daily prices and volume |
| `return_1d` | `close[t] / close[t-1] − 1` |
| `sma_10`, `sma_30` | 10- and 30-day simple moving averages of the close |
| `momentum_5` | `close[t] / close[t-5] − 1` |
| `volatility_20` | standard deviation of the last 20 daily returns |
| `rsi_14` | Wilder's 14-day Relative Strength Index |
| `volume_z_20` | `(volume − 20-day mean) / 20-day standard deviation` |
| `order_flow` | close location value `(2·close − high − low) / (high − low)` |
| `target` | 1 if `close[t+1] > close[t]` (verified on all 800 known rows) |

Hidden rows: Round 1 blanked `close`, `return_1d`, `sma_10`, `sma_30`, `momentum_5` and `target` for days 801–850. Round 2 revealed those days and blanked `close`, `return_1d` and `target` for days 851–900 (the moving averages and momentum stayed filled). Round 3 reused the Round 2 file.

## 2. Round 1: direction in a calm market

### 2.1 Hidden-row audit (`01_audit_round1.py`)

| Check | Result |
| --- | --- |
| `order_flow` = close location on known rows | max error 0.00005 |
| `order_flow` = **open** location on hidden rows | max error 0.00005 |
| `volatility_20` on hidden rows = volatility with hidden closes replaced by opens | max error 0.00001 |
| Wilder-RSI inversion on four known 50-day windows | exact (mean error 0.0000) |
| RSI-rebuilt hidden closes inside the day's low–high range | **14%** (true closes: 100%) |

Conclusion: in the test rows three columns meant something different from training. Only `open`, `high`, `low`, `volume` and `volume_z_20` were consistent.

### 2.2 Model search (`02_model_search.py`)

Broad 37-column technical feature set; expanding-window walk-forward, 30-day test blocks, one-day purge; 771 training rows, 18 folds, 521 out-of-sample days. Selection folds cover days 250–520; the rest were held back.

| Model | MCC, selection | MCC, held out | MCC, walk-forward | Balanced accuracy |
| --- | --- | --- | --- | --- |
| XGBoost | 0.133 | 0.123 | 0.128 | 0.564 |
| Logistic (C=0.01) | 0.097 | 0.147 | 0.121 | 0.560 |
| Random forest (depth 3) | 0.059 | 0.180 | 0.117 | 0.558 |
| Extra trees (depth 5) | 0.096 | 0.139 | 0.117 | 0.558 |
| HistGradientBoosting | **0.200** | **0.019** | 0.113 | 0.557 |
| Random forest (depth 5) | 0.111 | 0.107 | 0.109 | 0.555 |
| Extra trees (depth 3) | 0.089 | 0.131 | 0.109 | 0.555 |
| Logistic (C=0.1) | 0.067 | 0.147 | 0.105 | 0.553 |
| Logistic (C=0.03) | 0.067 | 0.123 | 0.094 | 0.547 |
| LightGBM | 0.119 | 0.059 | 0.090 | 0.545 |
| Logistic (C=1.0) | 0.023 | 0.155 | 0.086 | 0.543 |
| SVM (RBF) | 0.062 | 0.076 | 0.069 | 0.533 |

Gradient boosting looked best during selection and collapsed afterwards, a textbook overfit on ~800 noisy rows; regularised linear models were the most stable.

### 2.3 The final return-pattern model (`03_final_model_validation.py`)

Logistic regression (C = 0.05) on lags 0–4 of volatility-scaled returns, order flow and RSI change: walk-forward MCC **0.163**, balanced accuracy 0.581 over 542 days; bootstrap 95% CI [0.080, 0.251]; MCC positive in all ten 50-day blocks.

Sensitivity (walk-forward MCC):

| Lags | C = 0.02 | C = 0.05 | C = 0.1 | C = 0.3 |
| --- | --- | --- | --- | --- |
| 0–2 | 0.096 | 0.104 | 0.103 | 0.107 |
| 0–3 | 0.129 | 0.152 | 0.144 | 0.148 |
| 0–4 | 0.167 | **0.163** | 0.159 | 0.122 |
| 0–5 | 0.141 | 0.148 | 0.163 | 0.133 |
| 0–6 | 0.133 | 0.152 | 0.144 | 0.111 |

### 2.4 What we submitted (`04_round1_submission.py`)

Because the hidden rows were altered, the uploaded file used a logistic ensemble (3 values of C × 2 feature sets) on open-only features: walk-forward MCC 0.132, balanced accuracy 0.566. Public score 17.22. Section 5 shows what the return-pattern model would have scored with consistent inputs.

## 3. Round 2: the flash crash

### 3.1 Hidden-row audit (`05_audit_round2.py`)

`close[t] = 10 × (sma_10[t] − sma_10[t−1]) + close[t−10]` uses only day-*t* information.

| Check | Result |
| --- | --- |
| Rebuilt close inside the day's low–high range | 100% of hidden days |
| `rsi_14` recomputed from rebuilt closes | max error 0.00499 |
| `volatility_20` recomputed | max error 0.00005 |
| `order_flow` recomputed | max error 0.00302 |
| `momentum_5`, `sma_30` recomputed | max error 0.00007 |

The crash: price 107.95 → 59.49 over days 851–900; daily volatility 1.47% → 3.25%.

### 3.2 Model (`06_round2_submission.py`)

Return-pattern features plus volatility-scaled 20-day drawdown; logistic regression averaged over lags {4, 5, 6} × C {0.05, 0.1}; recency weights with a 150-day half-life. Trained on days 1–800 and tested on the revealed days 801–850 it scored MCC 0.206, balanced accuracy 0.604 (slightly optimistic: the drawdown feature was chosen on data that included those days).

Two versions: *static* (trained once on days 1–850) and *online* (retrained before each hidden day with only the outcomes known by that day). They agreed on 86% of days.

### 3.3 Uploads and what they revealed

| Upload | Public score | Interpretation |
| --- | --- | --- |
| Online model | 3.38 | far below the 16.67 coin-flip line: systematically wrong in the crash |
| Exact inverse of the online model | **29.96** | mirror image (3.38 + 29.96 = 2 × 16.67): the signal inverted rather than vanished |
| A separately trained teammate model | 13 | agreed with the inverted file on only 52% of days |

Because the platform keeps the latest upload, the last file replaced the 29.96; the final board records 6.30 for Round 2. The inversion was prompted by the leaderboard result; section 5 shows the market evidence behind it.

## 4. Round 3: the trading book

Scoring card: Sharpe ratio rewarded, maximum drawdown penalised at 2×, trading cost 0.001 per unit of position moved.

### 4.1 Position engine (`07_round3_positions.py`, `src/glacier/backtest.py`)

1. **Signal**: `clip(0.7 × trend + 0.3 × timing, −1, 1)`, where trend = `(SMA10/SMA30 − 1)/σ20` (halved if 5-day momentum disagrees, clipped to ±1) and timing = edge of the inverted Round 2 model (ignored below 0.02, scaled by 0.2).
2. **Volatility targeting**: multiply by `min(1, 1.2% / σ20)`.
3. **Drawdown guard**: halve the position while the book is more than 6% below its peak, using realised P&L up to day *t*.
4. **No-trade band**: keep yesterday's position unless the target moves by more than 0.12.

### 4.2 Design choices on days 300–849 (before the crash)

| Variant | Sharpe | Max drawdown | Total return | Turnover / day |
| --- | --- | --- | --- | --- |
| Trend only | 0.48 | −19.6% | +10.1% | 0.07 |
| First hedged blend (50/50, no guard, no band) | 0.56 | −12.4% | +12.9% | 0.49 |
| **Submitted v4 (70/30 + all controls)** | **0.64** | **−12.3%** | +11.1% | **0.13** |
| v4 without the drawdown guard | 0.55 | −17.6% | +12.0% | 0.22 |
| v4 balanced 50/50 | 0.74 | −8.9% | +12.7% | 0.29 |
| Timing-led 30/70 | 0.73 | −10.6% | +12.7% | 0.42 |

We shipped the trend-led version because a crash is a strong trend, while days 300–849 were calm. A second upload at one tenth of the size scored the same, confirming the scorer was scale-free (as Sharpe is).

## 5. Post-event analysis (`08_post_event_analysis.py`)

Computed after each round closed; nothing here informed a submission.

| Question | Answer |
| --- | --- |
| Round 1 uploads vs revealed labels | teammate model MCC −0.053; our open-feature v4 MCC −0.045 (both chance) |
| Return-pattern model with the true Round 1 inputs | MCC **+0.152**, balanced accuracy 0.576 on days 801–850 |
| Round 3 book over days 851–899 | **+16.3%** vs buy-and-hold −44.9%; Sharpe 5.84; max drawdown −2.2%; cost 0.33%; 16 trades; right direction on 63% of days |
| Correlation of next-day return with today's return | calm −0.11 → crash **+0.05** |
| … with yesterday's return | calm +0.13 → crash **−0.09** |

## 6. Metrics glossary

| Metric | Definition |
| --- | --- |
| Balanced accuracy | mean of the hit rates on UP days and on DOWN days; 0.5 = coin flip |
| MCC | Matthews correlation coefficient between predictions and outcomes, −1 to +1; 0 = no skill |
| Sharpe ratio | `mean(daily P&L) / std(daily P&L) × √252` |
| Max drawdown | largest fall of the book from a previous peak |
| Turnover | mean absolute change in position per day |
| Walk-forward validation | train on all earlier days, test on the next block, roll forward |
