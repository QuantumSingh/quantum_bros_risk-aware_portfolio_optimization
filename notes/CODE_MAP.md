# Code Map

## Original Paper Code

Original implementation is stored in:

`original_pennylane/qrl-dpo-public/`

## Main Entry Point

`MAIN.py`

This script runs the full portfolio optimization experiment.

## Data Input

`MAIN.py` loads price data from:

```python
price_data = pd.read_parquet('./data/price_data.parquet.gzip')
