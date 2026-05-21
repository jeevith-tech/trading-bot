# Institutional Crypto Bot Foundation

This workspace contains a risk-first decision engine for scanning liquid crypto perpetual markets. It is intentionally conservative: the scanner rejects low-quality markets, ranks candidate setups from 0 to 100, and only returns A+ opportunities above the configured threshold.

## What Is Implemented

- Multi-factor setup scoring with the requested weights:
  - HTF trend alignment: 20
  - Volume strength: 15
  - Market structure: 15
  - BTC correlation: 10
  - Liquidity conditions: 10
  - Open interest: 10
  - Volatility conditions: 10
  - Entry timing: 10
- Regime detection:
  - trending
  - range-bound
  - volatile breakout
  - compression
  - panic selloff
  - euphoria
  - news-driven
  - choppy
- Strategy detectors:
  - momentum breakout
  - mean reversion
  - trend continuation
  - liquidity sweep
- Risk controls:
- Paper replay scripts default to 10% requested risk per trade
  - Core scanner still supports lower volatility-adjusted risk for safer deployments
  - 3% daily loss kill switch
  - 8% weekly drawdown kill switch
  - stop after 3 consecutive losses
  - ATR-aware sizing through stop distance
- Market filters:
  - minimum daily volume
  - maximum spread
  - session liquidity
  - exchange health
  - news risk
- Trade logging model with entry reason, regime, position size, stop, market emotion, screenshot path, and result fields.
- Paper trading mode with fees, slippage, stop-losses, TP1/TP2 scale-outs, breakeven stop adjustment, mark-to-market PnL, and CSV trade export.
- Binance USDT-M futures public-data runner for replaying today's completed 15-minute candles.

## Run The Self-Test

```powershell
python .\self_test.py
```

## Calculate Today's Binance Paper PnL

```powershell
python .\scripts\paper_today_binance.py --max-symbols 140
```

Useful options:

```powershell
python .\scripts\paper_today_binance.py --max-symbols 40 --equity 100000 --risk-pct 10 --max-risk-pct 10 --min-score 85 --flat-at-end
```

For a small INR-denominated paper account:

```powershell
python .\scripts\paper_today_binance.py --max-symbols 140 --capital-inr 3000 --inr-per-usdt 95 --flat-at-end
```

To search conservative win-rate filters on today's data:

```powershell
python .\scripts\paper_today_binance.py --max-symbols 140 --capital-inr 3000 --inr-per-usdt 95 --optimize-winrate --min-optimizer-trades 5
```

## Free 24/7 Paper Mode Without Laptop

This repo includes a free scheduled paper runner:

```powershell
python .\scripts\live_paper_tick.py
```

Default live-paper settings:

- Starting capital: `3000 INR`
- Risk: `10%`
- Max trades: `1/day`
- Model: current v17 performance model
- Data source: Binance USDT-M public futures data
- Mode: paper only, no real orders

Outputs:

- State: `reports/live_paper/state.json`
- Daily PnL: `reports/live_paper/daily_summary.csv`
- Trades: `reports/live_paper/trades.csv`
- Open positions: `reports/live_paper/open_positions.csv`
- Simple dashboard text: `reports/live_paper/status.md`

For no-laptop operation with exact Binance data, run this repo on a free cloud VM in a Binance-supported region. Oracle Cloud Always Free is the most practical free option. On the VM, schedule this command every 15 minutes:

```bash
cd ~/trading-bot
python3 scripts/live_paper_tick.py
```

One-command Ubuntu VM installer:

```bash
wget -O - https://raw.githubusercontent.com/jeevith-tech/trading-bot/main/deploy/free-vm-install.sh | bash
```

Alternative if you already cloned the repo:

```bash
git clone https://github.com/jeevith-tech/trading-bot.git ~/trading-bot
cd ~/trading-bot
bash deploy/free-vm-install.sh
```

Check it later:

```bash
cd ~/trading-bot
bash deploy/free-vm-status.sh
```

GitHub Actions is included for manual diagnostics, but GitHub-hosted runners are often blocked by Binance with `HTTP 451`, so GitHub Actions is not reliable for Binance-accurate 24/7 tracking. If you use GitHub Actions anyway, open `.github/workflows/free-paper-trading.yml` from the Actions tab and run it manually to test connectivity.

For an elite one-trade-per-day validation:

```powershell
python .\scripts\paper_multi_day_binance.py --days 10 --max-symbols 20 --capital-inr 3000 --inr-per-usdt 95 --min-score 94 --max-positions 1 --one-trade-per-day --strategies momentum_breakout
```

This mode is intentionally allowed to skip days. A rule that forces exactly one trade every day is lower quality than a rule that waits for a rare clean setup.

For weekly validation across separate weeks:

```powershell
python .\scripts\paper_weekly_binance.py --weeks 4 --max-symbols 20 --capital-inr 3000 --inr-per-usdt 95 --compare
```

The weekly tester ranks variants by profitable weeks, then PnL, then win rate. Older Binance open-interest history may be unavailable; when that happens, OI-dependent score thresholds should not be compared directly against recent OI-enabled runs.

For explicit 10% risk weekly runs:

```powershell
python .\scripts\paper_weekly_binance.py --start-date 2026-05-01 --end-date 2026-05-20 --capital-inr 3000 --risk-pct 10 --max-risk-pct 10 --fixed-risk
```

For the fairest OI-free historical replay, rank the universe by quote volume inside the replay window and avoid current-volume bias:

```powershell
python .\scripts\paper_weekly_binance.py --weeks 3 --end-date 2026-04-21 --max-symbols 20 --candidate-symbols 40 --historical-volume-ranking --capital-inr 3000 --fixed-risk --skip-open-interest
```

Current strongest performance variant:

```powershell
python .\scripts\fast_filter_optimizer.py --start-date 2026-05-01 --end-date 2026-05-20 --lookback-days 30 --capital-inr 3000 --inr-per-usdt 95 --risk-pct 10 --max-risk-pct 10 --fixed-risk --max-symbols 20 --candidate-symbols 20 --historical-volume-ranking --strategies trend_continuation --directions long,short --thresholds 85 --btc-return-sets none --btc-21d-min-sets -10 --short-btc-21d-max-sets -10 --symbol-24h-max-sets 3.4 --symbol-atr-min-sets 0.48 --btc-72h-max-sets 3 --max-trades-per-day 1
```

January-improved v17 performance variant:

```powershell
python .\scripts\fast_filter_optimizer.py --start-date 2026-05-01 --end-date 2026-05-20 --lookback-days 30 --capital-inr 3000 --inr-per-usdt 95 --risk-pct 10 --max-risk-pct 10 --fixed-risk --max-symbols 20 --candidate-symbols 20 --historical-volume-ranking --strategies trend_continuation --directions long,short --thresholds 85 --btc-return-sets none --btc-21d-min-sets -10 --short-btc-21d-max-sets -10 --symbol-24h-max-sets 3.4 --symbol-atr-min-sets 0.48 --btc-72h-max-sets 3 --skip-long-btc21-min-sets 2 --skip-long-btc72-max-sets -1 --skip-long-symbol24-min-sets 1 --max-trades-per-day 1
```

Interpretation: use filtered longs when BTC's rolling 21-day return is above `-10%`; use shorts only in deeper bear regimes below `-10%`. Long entries also require BTC not to be overextended over 72h, symbol 24h return below `3.4%`, and 15m ATR at least `0.48%`. The v17 overlay skips late longs where BTC is still up over 21 days (`>= 2%`) but rolling over across 72h (`<= -1%`) while the coin is already up over 24h (`>= 1%`). This targets January's late-long failure without reducing the Feb-May comparison runs.

Defaults:

- Timezone: `Asia/Calcutta`
- Account: `100000 USDT`
- INR conversion override: `--inr-per-usdt`
- Paper risk: `10%` per trade
- Fee: `4 bps`
- Slippage: `2 bps`
- Minimum setup score: `85`
- Minimum 24h quote volume: `50000000 USDT`
- Trade CSV: `reports/binance_paper_today_trades.csv`

## Minimal Usage

```python
from institutional_bot import MarketScanner, MarketSnapshot, RiskState
from institutional_bot.models import TimeframeFrame

scanner = MarketScanner()
decision = scanner.scan(snapshots=[snapshot], risk_state=RiskState(equity=100_000))

for candidate, plan in decision.tradable:
    print(candidate.symbol, candidate.strategy, candidate.score, plan.quantity)
```

## Next Build Steps

1. Add exchange adapters for Binance, Bybit, OKX, Hyperliquid, and Coinbase Advanced.
2. Stream OHLCV, order book, funding, and open interest into `MarketSnapshot`.
3. Persist snapshots, candidates, decisions, and trade logs to PostgreSQL.
4. Add exchange-specific paper order books with limit-first queue assumptions and partial fill handling.
5. Add a dashboard for live ranked setups, exposure, drawdown, positions, and correlation warnings.
6. Build a walk-forward backtesting layer with fees, slippage, latency assumptions, and realistic fills.

This is not financial advice and is not ready for live capital until exchange connectivity, backtests, paper trading, monitoring, and operational safeguards are validated.
