# Week 5 — Stock Price Prediction using LSTM

An LSTM that predicts a stock's adjusted closing price and forecasts the
next 15 days, using log returns, MACD and RSI as features. Everything is in
[`Stock_Prediction_LSTM.ipynb`](Stock_Prediction_LSTM.ipynb).

## How to run

```bash
pip install yfinance tensorflow scikit-learn pandas numpy matplotlib
jupyter notebook Stock_Prediction_LSTM.ipynb
```

Run all cells. The notebook asks for a ticker (default `AAPL`), start and
end dates (defaults 2018-01-01 → 2026-01-01) and a timeframe
(daily / weekly / monthly). Defaults are used automatically when input is
unavailable (headless "run all"). Figures are saved as PNGs next to the
notebook.

## What it does

1. Downloads the data with `yfinance`.
2. Plots the adjusted close with Bollinger Bands, MACD and RSI.
3. Preprocessing: features z-scored with training statistics only, sliding
   windows of 60 days, chronological 80/20 split. The target is the next
   day's log return.
4. Model: `LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(1)`, Adam + MSE,
   early stopping.
5. Computes R² on train and test (plus a naive baseline) and a recursive
   15-day forecast.

## Why these indicators?

- **MACD (12, 26, 9)** — difference between a fast and a slow EMA; shows
  trend direction and momentum in one signal.
- **RSI (14)** — momentum oscillator between 0 and 100; above 70 =
  overbought, below 30 = oversold.
- Bollinger Bands are plotted for context but not used as a model input.

## Results (AAPL daily, 2018-01-01 → 2026-01-01)

| Metric | Value |
|---|---|
| R² (train) | 0.9981 |
| R² (test) | 0.9699 |
| R² (naive "tomorrow = today") | 0.9701 |

The test R² is essentially the same as the naive persistence baseline. That
is expected for one-step price prediction: daily prices are close to a
random walk, so a high R² mainly means the model tracks the price well, not
that it sees the future. Two things were fixed compared to an earlier
version of this notebook: the scalers are now fitted on training data only
(fitting them on the full series leaks test information and inflates R²),
and the model predicts returns instead of scaled price levels (levels can't
extrapolate when the test period trades above the training range).

The 15-day forecast is recursive, so it drifts with the recent trend rather
than reproducing day-to-day moves.
