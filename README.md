# Equity Investing

## Project Description

An equity investing system for research, backtesting, and automated trading.

## Directory Structure

The project is organized as follows:

```text
.
|-- alpha_models/
|-- data_preprocessing/
|   |-- fetch_market_data.py
|   |-- preprocess_market_data.py
|   `-- validate_data_preprocessing.ipynb
|-- feature_analysis/
|-- live_trading/
|-- model_backtest/
|-- REQUIREMENTS.md
|-- ARCHITECTURE.md
`-- DECISIONS.md
```

## Installation

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy pyarrow
```

Set Alpaca credentials if you want to fetch market data:

```bash
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
```

## Usage

Fetch historical trades:

```python
from data_preprocessing.fetch_market_data import fetch_historical_trades

trades = fetch_historical_trades(["AAPL", "MSFT"])
```

Build bars from trade data:

```python
from data_preprocessing.preprocess_market_data import get_volume_bars

bars = get_volume_bars(trades, threshold=10_000)
ohlcv = bars.ohlcv
```
