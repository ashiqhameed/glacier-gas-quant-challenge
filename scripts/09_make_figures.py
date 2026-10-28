"""Figures for the README (run 02, 07 and 08 first)."""
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from glacier.config import FIGURES_DIR, RESULTS_DIR, SUBMISSIONS_DIR, TRAIN_END_R1, TRAIN_END_R2 as N
from glacier.data import load_round, rebuild_closes_from_sma10
from glacier.features import crash_model_features
from glacier.models import fit_weighted_logistic

warnings.filterwarnings("ignore")
FIGURES_DIR.mkdir(exist_ok=True)
INK, GREY, TEAL, CORAL, BLUE, GOLD, VIOLET = "#1f2937", "#9ca3af", "#0f9d7a", "#e4572e", "#3b82f6", "#e0a100", "#7c5cd6"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "figure.dpi": 150, "savefig.bbox": "tight"})

d = load_round(2)
close = rebuild_closes_from_sma10(d)
days = d.timestamp.values
post = json.load(open(RESULTS_DIR / "post_event_analysis.json"))


def save(fig, name):
    fig.savefig(FIGURES_DIR / name, facecolor="white")
    plt.close(fig)
    print("wrote figures/" + name)


# 1. Round 3 book through the crash
pos = pd.read_csv(SUBMISSIONS_DIR / "round3" / "upload1_position_engine_v4.csv").signal.values[:49]
nxt = (close.shift(-1) / close - 1).values[N:N + 49]
pnl = pos * nxt - 0.001 * np.abs(np.diff(np.r_[0, pos]))
x = days[N + 1:N + 50]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
a1.plot(x, (np.cumprod(1 + pnl) - 1) * 100, color=TEAL, lw=2.4, label="Our book (submitted)")
a1.plot(x, (np.cumprod(1 + nxt) - 1) * 100, color=CORAL, lw=2.0, label="Buy and hold GLACIER_GAS")
a1.axhline(0, color=GREY, lw=0.8)
a1.set_ylabel("Cumulative return (%)")
a1.set_title("Round 3: our book vs buy-and-hold through the flash crash (days 851-899)")
m = post["round3_book_in_crash"]
a1.text(0.02, 0.06, f"Book {m['total_return']:+.1%}  |  Sharpe {m['sharpe']:.2f}  |  max drawdown {m['max_drawdown']:.1%}  |  "
        f"cost {m['total_cost']:.2%}  |  {int(m['trades'])} trades", transform=a1.transAxes, fontsize=9, color=INK)
a1.legend(frameon=False, loc="center left")
a2.bar(days[N:N + 49], pos, color=np.where(pos < 0, CORAL, TEAL), width=0.8)   # position decided on day t
a2.axhline(0, color=GREY, lw=0.8)
a2.set_ylabel("Position")
a2.set_xlabel("Day")
save(fig, "crash_equity_curve.png")

# 2. Price overview with the test windows
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(days, close, color=INK, lw=0.9, label="Close (days 851-900 rebuilt from SMA 10)")
ax.plot(days, d.sma_10, color=BLUE, lw=1.0, label="SMA 10")
ax.plot(days, d.sma_30, color=GOLD, lw=1.0, label="SMA 30")
ax.axvspan(801, 850, color=BLUE, alpha=0.08)
ax.axvspan(851, 900, color=CORAL, alpha=0.10)
ax.set_ylim(55, 127)
ax.text(798, 123, "Round 1 test (days 801-850)", ha="right", va="center", fontsize=8, color=BLUE)
ax.text(898, 117, "Rounds 2-3 test: flash crash (851-900)", ha="right", va="center", fontsize=8, color=CORAL)
ax.set_title("GLACIER_GAS: slow cycles for 800 days, then a 45% crash")
ax.set_xlabel("Day")
ax.set_ylabel("Price")
ax.legend(frameon=False, loc="lower left", fontsize=8)
save(fig, "price_overview.png")

# 3. Regime shift: volatility and the flipped autocorrelation
ac = post["autocorrelation_next_day"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
a1.plot(days, d.volatility_20 * 100, color=VIOLET, lw=1.0)
a1.axvspan(851, 900, color=CORAL, alpha=0.10)
a1.set_title("20-day volatility (%)")
a1.set_xlabel("Day")
labels = ["Today's return", "Yesterday's return"]
xx = np.arange(2)
calm = [ac["calm_days_1_850"]["lag0"], ac["calm_days_1_850"]["lag1"]]
crash = [ac["crash_days_851_899"]["lag0"], ac["crash_days_851_899"]["lag1"]]
a2.bar(xx - 0.18, calm, 0.36, color=BLUE, label="Calm (days 1-850)")
a2.bar(xx + 0.18, crash, 0.36, color=CORAL, label="Crash (days 851-899)")
for xi, v in zip(np.r_[xx - 0.18, xx + 0.18], calm + crash):
    a2.text(xi, v + (0.008 if v >= 0 else -0.008), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
a2.axhline(0, color=GREY, lw=0.8)
a2.set_ylim(-0.14, 0.17)
a2.set_xticks(xx, labels)
a2.set_title("Correlation with next-day return")
a2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
fig.suptitle("Concept drift: volatility doubled and the return pattern flipped sign", fontweight="bold", y=1.03)
save(fig, "regime_shift.png")

# 4. Model comparison: walk-forward MCC and selection vs held-out MCC
ms = pd.read_csv(RESULTS_DIR / "model_search.csv").sort_values("mcc_walk_forward")
final = json.load(open(RESULTS_DIR / "final_model_validation.json"))["walk_forward"]["mcc"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
names = list(ms.model) + ["Final: compact logistic\n(return/flow/RSI lags)"]
vals = list(ms.mcc_walk_forward) + [final]
a1.barh(names, vals, color=[GREY] * len(ms) + [TEAL])
for i, v in enumerate(vals):
    a1.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
a1.set_title("Walk-forward MCC (500+ out-of-sample days)")
a1.set_xlim(0, max(vals) * 1.2)
ms2 = ms.sort_values("mcc_selection")
y2 = np.arange(len(ms2))
a2.hlines(y2, ms2.mcc_selection, ms2.mcc_holdout, color=GREY, lw=1.5)
a2.scatter(ms2.mcc_selection, y2, facecolor="white", edgecolor=INK, zorder=3, label="Selection folds (days 250-520)")
a2.scatter(ms2.mcc_holdout, y2, color=[CORAL if m == "HistGradientBoosting" else BLUE for m in ms2.model], zorder=3,
           label="Held-out folds (days 520-771)")
a2.set_yticks(y2, ms2.model)
a2.axvline(0, color=GREY, lw=0.8)
a2.set_title("Selection vs held-out: gradient boosting overfit")
a2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
fig.tight_layout()
save(fig, "model_comparison.png")

# 5. Feature importance of the shipped model (standardised logistic coefficients)
G = crash_model_features(d, close, 5)
label = d.target
train = [i for i in range(N) if G.iloc[i].notna().all() and not np.isnan(label.iloc[i])]
coef = pd.Series(fit_weighted_logistic(G, label, train, C=0.1, half_life=150).named_steps["logisticregression"].coef_[0], index=G.columns)
pretty = {"z": "Vol-scaled return", "f": "Order flow", "rc": "RSI change"}


def nice(k):
    if k == "dd20z":
        return "20-day drawdown / vol"
    for p in ("rc", "z", "f"):
        if k.startswith(p):
            lag = int(k[len(p):])
            return f"{pretty[p]} ({'today' if lag == 0 else f't-{lag}'})"


top = coef.reindex(coef.abs().sort_values().index)[-10:]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.barh([nice(k) for k in top.index], top.values, color=np.where(top.values > 0, TEAL, CORAL))
ax.axvline(0, color=GREY, lw=0.8)
ax.set_title("What the model learned: standardised logistic coefficients (top 10)")
ax.set_xlabel("Coefficient (positive = more likely UP tomorrow)")
save(fig, "feature_importance.png")

# 6. Walk-forward validation scheme
fig, ax = plt.subplots(figsize=(9, 2.8))
for i, s in enumerate([250, 400, 550, 700]):
    yy = 3 - i
    ax.add_patch(Rectangle((0, yy), s - 1, 0.6, color=BLUE, alpha=0.75))
    ax.add_patch(Rectangle((s, yy), 50, 0.6, color=CORAL))
    ax.text(s + 60, yy + 0.3, f"fold {i + 1}", va="center", fontsize=8, color=INK)
ax.add_patch(Rectangle((800, -0.9), 50, 0.6, color=GOLD))
ax.text(860, -0.6, "organisers' hidden test window", va="center", fontsize=8, color=INK)
ax.set_xlim(0, 1000)
ax.set_ylim(-1.2, 4.1)
ax.set_yticks([])
ax.set_xlabel("Day")
ax.set_title("Expanding-window walk-forward validation")
ax.text(0, 3.8, "Train on all earlier days (blue), test on the next block (red), roll forward. Never shuffled.", fontsize=9, color=INK)
ax.spines["left"].set_visible(False)
save(fig, "walk_forward_validation.png")

# 7. Round 3 design choices: Sharpe vs drawdown on pre-crash data
bt = pd.read_csv(RESULTS_DIR / "round3_backtests.csv")
fig, ax = plt.subplots(figsize=(7.5, 4.3))
for _, r in bt.iterrows():
    shipped = r.variant.startswith("Submitted")
    ax.scatter(-r.max_drawdown * 100, r.sharpe, s=40 + 300 * r.turnover_per_day, color=TEAL if shipped else GREY,
               edgecolor=INK if shipped else "none", zorder=3)
    ax.annotate(r.variant.split(":")[0].replace(" (50/50, no guard, no band)", ""), (-r.max_drawdown * 100, r.sharpe),
                xytext=(7, 4), textcoords="offset points", fontsize=8, color=INK)
ax.set_xlabel("Max drawdown (%) - lower is better")
ax.set_ylabel("Annualised Sharpe")
ax.set_title("Round 3 design choices on pre-crash data (bubble size = turnover)")
ax.set_xlim(5, 24)
save(fig, "round3_design_choices.png")
