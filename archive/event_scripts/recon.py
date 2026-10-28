"""Reconstruct unknown closes from rsi_14 (Wilder RSI, verified to match on rows 1-800).
Only uses information available at time t (rsi_14[t] and earlier state) - no lookahead."""
import numpy as np
import pandas as pd


def wilder_state(c):
    dl = c.diff()
    g, l = dl.clip(lower=0), (-dl).clip(lower=0)
    return g.ewm(alpha=1 / 14, adjust=False).mean().values, l.ewm(alpha=1 / 14, adjust=False).mean().values


def reconstruct(close_known, rsi_future, n_known):
    c = pd.Series(close_known[:n_known])
    AG, AL = wilder_state(c)
    ag, al, last, out = AG[-1], AL[-1], c.iloc[-1], []
    for r in rsi_future:
        rs = r / (100 - r)
        al_u = 13 * al / 14
        ag_u = rs * al_u
        g = 14 * ag_u - 13 * ag          # gain implied if today was an up day
        ag_d = 13 * ag / 14
        al_d = ag_d / rs
        l = 14 * al_d - 13 * al          # loss implied if today was a down day
        cand = []
        if g >= 0:
            cand.append((g, ag_u, al_u, +1))
        if l >= 0:
            cand.append((l, ag_d, al_d, -1))
        if not cand:
            cand = [(0.0, 13 * ag / 14, 13 * al / 14, 0)]
        m, ag, al, s = min(cand, key=lambda x: x[0])
        last = last + s * m
        out.append(last)
    return np.array(out)


if __name__ == "__main__":
    d = pd.read_excel(r"data/GLACIER_GAS_C02_round1.csv.xlsx")
    c = d.close.values
    for start in (300, 500, 700, 750):
        rec = reconstruct(c, d.rsi_14.values[start:start + 50], start)
        true = c[start:start + 50]
        prev = np.r_[c[start - 1], true][:-1]
        prev_r = np.r_[c[start - 1], rec][:-1]
        rr, tr = (rec - prev_r) / prev_r, (true - prev) / prev
        print(start, "close MAE", np.abs(rec - true).mean().round(3), "ret corr", np.corrcoef(rr, tr)[0, 1].round(3),
              "sign acc", (np.sign(rr) == np.sign(tr)).mean())
