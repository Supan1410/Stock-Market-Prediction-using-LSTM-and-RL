# Week 7–8 — Final Project: Stock Prediction with LSTM and RL

Three models predict a stock's next adjusted closing price from the same
features (log return, MACD, RSI) and are compared on **R²** and on
**returns/risk** with a simple trading backtest:

- **LSTM** — supervised benchmark, predicts the next-day log return.
- **DQN** — RL agent whose action is the predicted return (7 quantile
  buckets).
- **DDPG** — the same with a continuous action.

All models also produce the required 15-day forecast, and the pipeline is
repeated on four tickers (AAPL, GOOGL, MSFT, TSLA). A naive persistence
baseline ("tomorrow = today") is always included.

Everything is in
[`LSTM_RL_Stock_Prediction.ipynb`](LSTM_RL_Stock_Prediction.ipynb).

## How to run

```bash
pip install yfinance tensorflow scikit-learn pandas numpy matplotlib gymnasium
jupyter notebook LSTM_RL_Stock_Prediction.ipynb
```

Takes ~6–10 min on CPU. The notebook asks for ticker / dates / timeframe
(defaults: `AAPL`, `2018-01-01`, today, `daily`; used automatically when
input is unavailable). Data is fetched live from Yahoo Finance, so numbers
shift slightly when re-run on a later date. Figures are saved as PNGs.

## Data & method

Executed run: AAPL daily, 2018-01-03 → 2026-07-07 (2,137 rows). Features:
log return, **MACD (12, 26, 9)** for trend/momentum and **RSI (14)** for
overbought/oversold conditions; Bollinger Bands are plotted for context.
Split: chronological, last ~6 months held out for testing, final 30 trading
days (the "past 6 weeks") reported separately. Scalers are fitted on the
training part only.

- **LSTM:** `LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(1)`, 60-day
  windows, target = next-day log return, price rebuilt as
  `previous close × exp(predicted return)`.
- **RL environment** (`gymnasium`): state = last 10 days of features,
  action = predicted next-day return, reward = −|prediction error|, γ = 0
  since the reward is immediate. DQN picks among 7 quantile buckets, DDPG
  outputs a continuous return.
- **Backtest** (returns & risk): hold the stock on days the model predicts
  a positive return, otherwise stay in cash — no shorting, no transaction
  costs. Compared against buy & hold on Sharpe ratio, returns, volatility
  and max drawdown.

All models predict returns rather than price levels — an earlier version
predicted scaled levels and broke when the test period traded above the
training range.

## Results

**Prediction accuracy — AAPL:**

| Model | R² (6-month holdout) | R² (final 30 days) | MAE $ |
|---|---|---|---|
| LSTM | 0.9449 | 0.6487 | 3.43 |
| DQN | 0.9199 | 0.5272 | 4.22 |
| DDPG | 0.9259 | 0.5439 | 4.09 |
| Persistence (naive) | 0.9450 | 0.6463 | 3.41 |

**R² on the holdout, all four tickers:**

| Ticker | LSTM | DQN | DDPG | Persistence |
|---|---|---|---|---|
| AAPL | 0.9449 | 0.9199 | 0.9259 | 0.9450 |
| GOOGL | 0.9560 | 0.9354 | 0.9325 | 0.9577 |
| MSFT | 0.9102 | 0.8971 | 0.8874 | 0.9112 |
| TSLA | 0.8049 | 0.7603 | 0.7341 | 0.8085 |

**Returns & risk (long/flat backtest), AAPL holdout:**

| Strategy | Cum. return % | Sharpe | Max drawdown % | Hit rate % | Trades |
|---|---|---|---|---|---|
| LSTM | 4.9 | 0.59 | −10.3 | 48.4 | 9 |
| DQN | 8.6 | 0.69 | −10.8 | 52.4 | 39 |
| DDPG | −9.1 | −0.97 | −15.5 | 46.8 | 61 |
| Buy & hold | 14.8 | 1.04 | −12.7 | 52.4 | 0 |

**Sharpe on the holdout, all four tickers:**

| Ticker | LSTM | DQN | DDPG | Buy & hold |
|---|---|---|---|---|
| AAPL | 0.59 | 0.69 | −0.97 | 1.04 |
| GOOGL | 0.49 | 0.44 | 0.57 | 1.00 |
| MSFT | −0.64 | −1.06 | −0.45 | −1.12 |
| TSLA | −0.19 | −0.35 | 0.05 | −0.38 |

## Comments on the results

- **On R², everything sits at the persistence baseline.** Daily prices are
  close to a random walk, so one-step price R² mostly rewards tracking —
  even "tomorrow = today" scores ~0.94 on AAPL. The LSTM is the best learned
  model on every ticker; DQN and DDPG swap places depending on the stock.
  (A model far *above* persistence usually means data leakage, not skill.)
- **The backtest separates the models much more than R².** On AAPL, DQN
  made 8.6 % (Sharpe 0.69) vs LSTM's 4.9 % (0.59), while DDPG lost 9.1 %
  (−0.97) despite a similar R². Hit rates are all near 50 % — the P&L
  depends on *which* days a model gets right, not its average error.
- **Risk:** the long/flat strategies run at lower volatility than
  buy & hold and lose less on the tickers that fell (MSFT, TSLA), but they
  miss up-days in rising markets, so none beats buy & hold there. With
  39–61 trades, DQN and DDPG would also suffer most from transaction costs.
- Scalers fitted on the full series silently inflate R² — this and the
  level-vs-return target were the two main bugs fixed relative to the
  Week 5 version.

