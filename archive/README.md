# Event scripts (archive)

The scripts exactly as written during the timed event (15–16 October 2026), kept for the record. Local file paths were replaced with `data/`; nothing else was changed.

They are superseded by the [`glacier`](../src/glacier) package and the numbered [`scripts/`](../scripts), which reproduce the same results and submission files with tests and clear structure. Roughly in the order they were written:

| Script | Purpose during the event |
| --- | --- |
| `starter.py`, `submit.py` | templates prepared before the data was released |
| `round1.py` | first Round 1 baseline (open-based features, logistic + random forest) |
| `recon.py` | Wilder-RSI inversion used to rebuild hidden closes |
| `search_r1.py`, `final_r1.py` | 12-model walk-forward search and ensemble |
| `ar_r1.py`, `check_r1.py`, `final_r1_v3.py` | the return-pattern model and its robustness checks |
| `r1_v4.py`, `final_r1_v4.py` | Round 1 v4 after the hidden-row audit (open-only features) |
| `r2_recon.py`, `r2_lib.py`, `final_r2.py` | Round 2 close reconstruction, features and models |
| `r2_adaptive.py` | short-window adaptive alternative for the crash |
| `r3.py`, `r3_v2.py`, `r3_v4.py` | Round 3 position engines; `r3_v4.py` produced the submitted book |
