# Stock Market Prediction using LSTM and RL

A Seasons of Code project that builds up from Python fundamentals to a final
project predicting stock prices with an LSTM and reinforcement-learning agents.
Each week is a self-contained set of Jupyter notebooks; the technical content
of all seven weeks is summarized below.

## Stack

Python · NumPy · Pandas · Matplotlib · scikit-learn · TensorFlow/Keras ·
PyTorch (Week 4 CNN) · Gymnasium · yfinance

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow torch torchvision gymnasium yfinance jupyter
```

## Week-by-week

**Week 1 — Python & data stack.** NumPy, Pandas and Matplotlib exercises plus
functions/control-flow practice (guessing game). Foundations for numerical
work and plotting.

**Week 2 — Market data with yfinance.** A `download_historical_data` helper to
pull OHLCV history for a ticker between two dates, plus closing-price
visualization.

**Week 3 — Classical ML.** Linear regression, logistic regression and K-Means
clustering assignments, backed by notes on neural networks, gradient descent
and backpropagation.

**Week 4 — Deep learning from scratch.** An RNN and LSTM implemented forward
and backward in pure NumPy (`rnn_utils.py`), plus a CNN for handwritten-digit
recognition in PyTorch. This is the theoretical basis for the LSTM used later.

**Week 5 — LSTM stock prediction.** An LSTM predicting a stock's next-day
**log return** (price rebuilt from it), using MACD (12, 26, 9) and RSI (14) as
features. Architecture `LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(1)`,
60-day sliding windows, chronological 80/20 split, Adam + MSE with early
stopping. Scalers are fitted on training data only, and a 15-day recursive
forecast is produced. Key lesson: test R² ≈ the naive "tomorrow = today"
baseline because daily prices are near a random walk — a high R² means good
tracking, not foresight. (AAPL: R² test 0.97 vs naive 0.97.)

**Week 6 — Reinforcement learning (Cart-Pole DQN).** A Double-DQN agent
balancing the inverted pendulum in Gymnasium's `CartPole-v1`. Q-network MLP
4 → 128 → 128 → 2, Huber loss, γ = 0.99, experience replay, soft-updated
target network, ε-greedy 1.0 → 0.01. DQN (discrete actions) is used instead of
DDPG; the Double-DQN target fixes overestimation bias that plateaued reward at
~320. Reaches a perfect greedy policy (100/100 episodes at the 500-step cap →
solved) and recovers from ±0.3 rad/s disturbance kicks.

**Weeks 7–8 — Final project: LSTM vs RL.** Three models predict the next-day
return from the same features (log return, MACD, RSI) and are compared on
**R²** and on a **long/flat trading backtest** (Sharpe, return, drawdown):

- **LSTM** — supervised benchmark.
- **DQN** — RL agent choosing among 7 quantile return buckets.
- **DDPG** — same, with a continuous action.

All three also produce the 15-day forecast, run across four tickers
(AAPL, GOOGL, MSFT, TSLA), against a naive persistence baseline.

### Final results (AAPL, 6-month holdout)

| Model | R² | Backtest return % | Sharpe |
|---|---|---|---|
| LSTM | 0.9449 | 4.9 | 0.59 |
| DQN | 0.9199 | 8.6 | 0.69 |
| DDPG | 0.9259 | −9.1 | −0.97 |
| Persistence / Buy & hold | 0.9450 | 14.8 | 1.04 |

**Takeaways:** on R² every model sits at the persistence baseline (one-step
price prediction mostly rewards tracking), so the backtest is what separates
them — models with near-identical R² diverge sharply in P&L depending on
*which* days they call right. The two bugs that mattered most were fitting
scalers on the full series (leaks test data, inflates R²) and predicting
scaled price levels instead of returns (breaks when the test period trades
above the training range).

## Running

Each week's notebooks run standalone (`jupyter notebook <file>.ipynb`, run all
cells). The Week 5 and 7–8 notebooks fetch data live from Yahoo Finance and
prompt for ticker / dates / timeframe, falling back to defaults (AAPL, daily)
when run headless; figures are saved as PNGs next to each notebook.
