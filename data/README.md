# Data

The competition data belongs to the organisers (Datazen, Abhiyantriki 2026) and is not redistributed in this repository.

To rerun the pipeline, place the round files here (or point `GLACIER_DATA_DIR` at another folder):

| File | Contents |
| --- | --- |
| `GLACIER_GAS_C02_round1.csv.xlsx` | days 1–850; close, returns, moving averages, momentum and target blank for days 801–850 |
| `GLACIER_GAS_C02_round2.csv` | days 1–900; days 801–850 revealed, close, return and target blank for days 851–900 |
| `GLACIER_GAS_C02_round3.csv` | identical to the Round 2 file (optional; Round 2 is used if missing) |

Columns: `timestamp, date, asset_id|stock_id, open, high, low, close, volume, return_1d, sma_10, sma_30, momentum_5, volatility_20, rsi_14, volume_z_20, order_flow, target`. See [`docs/methodology.md`](../docs/methodology.md#1-data) for definitions.

The unit tests (`pytest`) use synthetic data and need none of these files.
