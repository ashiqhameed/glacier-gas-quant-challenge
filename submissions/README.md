# Submissions

The files uploaded during the event, kept unchanged. The pipeline scripts regenerate each of our own files and check that the result is identical.

| Round | File | What it is | Public score |
| --- | --- | --- | --- |
| 1 | `round1/upload1_teammate_model.csv` | a separately trained teammate model, using the altered hidden columns | 13 |
| 1 | `round1/upload2_open_features_v4.csv` | logistic ensemble on open/high/low/volume features (`04_round1_submission.py`) | 17.22 |
| 2 | `round2/upload1_online_model.csv` | causal online logistic ensemble (`06_round2_submission.py`) | 3.38 |
| 2 | `round2/upload2_regime_inverted.csv` | the same predictions inverted for the crash regime | **29.96** |
| 2 | `round2/upload3_teammate_model.csv` | a separately trained teammate model | 13 |
| 3 | `round3/upload1_position_engine_v4.csv` | position engine v4 (`07_round3_positions.py`) | **100** (private 73.08) |
| 3 | `round3/upload2_v4_tenth_size.csv` | v4 at one tenth of the size, to test whether the scorer depends on size | unchanged |

Formats: Rounds 1–2 `timestamp,prediction,confidence` (prediction 1 = up); Round 3 `timestamp,signal` with signal in [−1, 1]. The leaderboard counted the latest upload in each round; the final standings used private scores (Round 1: 12.89, Round 2: 6.30, Round 3: 73.08).
