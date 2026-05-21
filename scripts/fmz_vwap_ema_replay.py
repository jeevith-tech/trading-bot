from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from institutional_bot.binance_data import BinanceFuturesClient, Candle
from scripts.paper_today_binance import load_timezone


@dataclass
class Position:
    direction: str
    entry_time: datetime
    entry: float
    stop: float
    tp2: float
    tp3: float
    quantity: float
    remaining: float
    realized_pnl: float
    fees_paid: float
    tp2_hit: bool = False


@dataclass(frozen=True)
class Trade:
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry: float
    exit_price: float
    quantity: float
    realized_pnl: float
    fees_paid: float
    exit_reason: str


def main() -> None:
    args = parse_args()
    tz = load_timezone(args.timezone)
    start_local = datetime.combine(date.fromisoformat(args.start_date), datetime.min.time(), tzinfo=tz)
    end_local = datetime.combine(date.fromisoformat(args.end_date) + timedelta(days=1), datetime.min.time(), tzinfo=tz)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)
    lookback_start = start_utc - timedelta(days=args.lookback_days)

    client = BinanceFuturesClient(timeout=args.timeout)
    candles = client.klines(args.symbol, args.interval, lookback_start, end_utc)
    candles = [candle for candle in candles if candle.open_time + interval_delta(args.interval) <= end_utc]
    result = replay(candles, start_utc, end_utc, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_trades(output_dir / "fmz_vwap_ema_trades.csv", result["trades"])
    write_summary(output_dir / "fmz_vwap_ema_summary.csv", result, args)
    print_report(result, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the public FMZ VWAP EMA Breakout strategy.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--timezone", default="Asia/Calcutta")
    parser.add_argument("--start-date", default="2026-05-01")
    parser.add_argument("--end-date", default="2026-05-20")
    parser.add_argument("--capital-inr", type=float, default=3000)
    parser.add_argument("--inr-per-usdt", type=float, default=95.0)
    parser.add_argument("--risk-pct", type=float, default=10.0)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument(
        "--exit-mode",
        choices=("pine_default", "split_50_50"),
        default="pine_default",
        help="pine_default exits 100%% at the first FMZ strategy.exit target; split_50_50 treats the two targets as scaled exits.",
    )
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--output-dir", default="reports/fmz_vwap_ema_may")
    return parser.parse_args()


def replay(candles: list[Candle], start_utc: datetime, end_utc: datetime, args: argparse.Namespace) -> dict[str, object]:
    equity = args.capital_inr / args.inr_per_usdt
    starting_equity = equity
    position: Position | None = None
    trades: list[Trade] = []
    pending_signal: dict[str, float | str | datetime] | None = None

    for idx, candle in enumerate(candles):
        if idx < 220:
            continue
        if candle.open_time < start_utc:
            continue
        if candle.open_time >= end_utc:
            break

        if pending_signal and position is None:
            position, equity = open_position(pending_signal, candle, equity, args)
            pending_signal = None

        if position is not None:
            position, closed_trade, equity = update_position(position, candle, equity, args)
            if closed_trade:
                trades.append(closed_trade)

        if position is None and idx + 1 < len(candles):
            signal = signal_at(candles, idx)
            if signal is not None:
                pending_signal = signal

    if position is not None:
        last = next(c for c in reversed(candles) if c.open_time < end_utc)
        closed_trade, equity = close_position(position, last.open_time, last.close, equity, "month_end", args)
        trades.append(closed_trade)

    wins = sum(1 for trade in trades if trade.realized_pnl > 0)
    losses = sum(1 for trade in trades if trade.realized_pnl <= 0)
    gross_profit = sum(trade.realized_pnl for trade in trades if trade.realized_pnl > 0)
    gross_loss = abs(sum(trade.realized_pnl for trade in trades if trade.realized_pnl < 0))
    return {
        "starting_equity": starting_equity,
        "ending_equity": equity,
        "pnl_usdt": equity - starting_equity,
        "pnl_inr": (equity - starting_equity) * args.inr_per_usdt,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades) * 100 if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
    }


def signal_at(candles: list[Candle], idx: int) -> dict[str, float | str | datetime] | None:
    close_values = [c.close for c in candles[: idx + 1]]
    ema10 = ema(close_values, 10)
    ema20 = ema(close_values, 20)
    ema200 = ema(close_values, 200)
    vwap = session_vwap(candles, idx)
    atr_value = atr(candles[: idx + 1], 14)
    candle = candles[idx]
    swing_low = min(c.low for c in candles[idx - 9 : idx + 1])
    swing_high = max(c.high for c in candles[idx - 9 : idx + 1])

    long_condition = (
        candle.close > vwap
        and candle.close > ema200
        and candle.close > ema10
        and candle.close > ema20
        and candle.close > candle.open
        and vwap > ema200
        and ema10 > ema20
        and ema20 > vwap
    )
    if long_condition:
        stop = swing_low - atr_value
        risk = candle.close - stop
        if risk > 0:
            return {
                "direction": "long",
                "time": candle.open_time,
                "entry_reference": candle.close,
                "stop": stop,
                "tp2": candle.close + risk * 2,
                "tp3": candle.close + risk * 3,
            }

    short_condition = (
        candle.close < vwap
        and candle.close < ema200
        and candle.close < ema10
        and candle.close < ema20
        and candle.close < candle.open
        and vwap < ema200
        and ema10 < ema20
        and ema20 < vwap
    )
    if short_condition:
        stop = swing_high + atr_value
        risk = stop - candle.close
        if risk > 0:
            return {
                "direction": "short",
                "time": candle.open_time,
                "entry_reference": candle.close,
                "stop": stop,
                "tp2": candle.close - risk * 2,
                "tp3": candle.close - risk * 3,
            }
    return None


def open_position(
    signal: dict[str, float | str | datetime],
    candle: Candle,
    equity: float,
    args: argparse.Namespace,
) -> tuple[Position, float]:
    direction = str(signal["direction"])
    entry = candle.open * (1 + args.slippage_bps / 10_000) if direction == "long" else candle.open * (1 - args.slippage_bps / 10_000)
    stop = float(signal["stop"])
    stop_distance = abs(entry - stop)
    account_risk = equity * args.risk_pct / 100
    quantity = account_risk / stop_distance
    fee = abs(entry * quantity) * args.fee_bps / 10_000
    return (
        Position(
            direction=direction,
            entry_time=candle.open_time,
            entry=entry,
            stop=stop,
            tp2=float(signal["tp2"]),
            tp3=float(signal["tp3"]),
            quantity=quantity,
            remaining=quantity,
            realized_pnl=-fee,
            fees_paid=fee,
        ),
        equity - fee,
    )


def update_position(
    position: Position,
    candle: Candle,
    equity: float,
    args: argparse.Namespace,
) -> tuple[Position | None, Trade | None, float]:
    if position.direction == "long":
        stop_hit = candle.low <= position.stop
        tp2_hit = candle.high >= position.tp2
        tp3_hit = candle.high >= position.tp3
    else:
        stop_hit = candle.high >= position.stop
        tp2_hit = candle.low <= position.tp2
        tp3_hit = candle.low <= position.tp3

    if stop_hit:
        trade, equity = close_position(position, candle.open_time, position.stop, equity, "stop_loss", args)
        return None, trade, equity
    if tp2_hit and args.exit_mode == "pine_default":
        trade, equity = close_position(position, candle.open_time, position.tp2, equity, "tp2", args)
        return None, trade, equity
    if tp2_hit and not position.tp2_hit:
        equity = close_fraction(position, candle.open_time, position.tp2, 0.5, equity, args)
        position.tp2_hit = True
    if tp3_hit:
        trade, equity = close_position(position, candle.open_time, position.tp3, equity, "tp3", args)
        return None, trade, equity
    return position, None, equity


def close_fraction(position: Position, timestamp: datetime, price: float, fraction: float, equity: float, args: argparse.Namespace) -> float:
    qty = min(position.remaining, position.quantity * fraction)
    fill = price * (1 - args.slippage_bps / 10_000) if position.direction == "long" else price * (1 + args.slippage_bps / 10_000)
    pnl = (fill - position.entry) * qty if position.direction == "long" else (position.entry - fill) * qty
    fee = abs(fill * qty) * args.fee_bps / 10_000
    position.remaining -= qty
    position.realized_pnl += pnl - fee
    position.fees_paid += fee
    return equity + pnl - fee


def close_position(position: Position, timestamp: datetime, price: float, equity: float, reason: str, args: argparse.Namespace) -> tuple[Trade, float]:
    equity = close_fraction(position, timestamp, price, 1.0, equity, args)
    fill = price * (1 - args.slippage_bps / 10_000) if position.direction == "long" else price * (1 + args.slippage_bps / 10_000)
    return (
        Trade(
            direction=position.direction,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry=position.entry,
            exit_price=fill,
            quantity=position.quantity,
            realized_pnl=position.realized_pnl,
            fees_paid=position.fees_paid,
            exit_reason=reason,
        ),
        equity,
    )


def ema(values: list[float], period: int) -> float:
    alpha = 2 / (period + 1)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1 - alpha) * result
    return result


def atr(candles: list[Candle], period: int) -> float:
    true_ranges = []
    for idx in range(1, len(candles)):
        true_ranges.append(
            max(
                candles[idx].high - candles[idx].low,
                abs(candles[idx].high - candles[idx - 1].close),
                abs(candles[idx].low - candles[idx - 1].close),
            )
        )
    window = true_ranges[-period:]
    return sum(window) / len(window)


def session_vwap(candles: list[Candle], idx: int) -> float:
    session_date = candles[idx].open_time.date()
    numerator = 0.0
    denominator = 0.0
    for candle in reversed(candles[: idx + 1]):
        if candle.open_time.date() != session_date:
            break
        typical = (candle.high + candle.low + candle.close) / 3
        numerator += typical * candle.volume
        denominator += candle.volume
    return numerator / denominator if denominator else candles[idx].close


def interval_delta(interval: str) -> timedelta:
    return {
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "2h": timedelta(hours=2),
    }[interval]


def write_trades(path: Path, trades: list[Trade]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["direction", "entry_time", "exit_time", "entry", "exit_price", "quantity", "realized_pnl", "fees_paid", "exit_reason"])
        for trade in trades:
            writer.writerow(
                [
                    trade.direction,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat(),
                    f"{trade.entry:.8f}",
                    f"{trade.exit_price:.8f}",
                    f"{trade.quantity:.8f}",
                    f"{trade.realized_pnl:.6f}",
                    f"{trade.fees_paid:.6f}",
                    trade.exit_reason,
                ]
            )


def write_summary(path: Path, result: dict[str, object], args: argparse.Namespace) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "start_date", "end_date", "interval", "risk_pct", "exit_mode", "trades", "win_rate", "pnl_usdt", "pnl_inr", "ending_equity_usdt"])
        writer.writerow(
            [
                args.symbol,
                args.start_date,
                args.end_date,
                args.interval,
                args.risk_pct,
                args.exit_mode,
                len(result["trades"]),
                f"{result['win_rate']:.2f}",
                f"{result['pnl_usdt']:.6f}",
                f"{result['pnl_inr']:.2f}",
                f"{result['ending_equity']:.6f}",
            ]
        )


def print_report(result: dict[str, object], args: argparse.Namespace) -> None:
    print()
    print("FMZ VWAP EMA Breakout replay")
    print(f"Symbol:          {args.symbol}")
    print(f"Interval:        {args.interval}")
    print(f"Exit mode:       {args.exit_mode}")
    print(f"Period:          {args.start_date} to {args.end_date}")
    print(f"Trades:          {len(result['trades'])}")
    print(f"Win rate:        {result['win_rate']:.1f}%")
    print(f"Profit factor:   {result['profit_factor']:.2f}")
    print(f"Starting equity: {result['starting_equity']:.2f} USDT")
    print(f"Ending equity:   {result['ending_equity']:.2f} USDT")
    print(f"PnL:             {result['pnl_usdt']:.2f} USDT / {result['pnl_inr']:.2f} INR")


if __name__ == "__main__":
    main()
