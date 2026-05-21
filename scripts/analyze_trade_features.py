from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from institutional_bot.binance_data import BinanceFuturesClient, Candle
from institutional_bot.indicators import atr, ema, rsi, slope, volume_expansion
from scripts.paper_today_binance import completed, index_at_or_before


@dataclass(frozen=True)
class TradeRow:
    source: str
    date: str
    symbol: str
    strategy: str
    direction: str
    score: float
    opened_at: datetime
    closed_at: datetime
    realized_pnl: float
    exit_reason: str


def main() -> None:
    args = parse_args()
    trades = read_trades([Path(path) for path in args.trade_csv])
    if not trades:
        raise RuntimeError("no trades found")

    symbols = sorted({trade.symbol for trade in trades} | {"BTCUSDT"})
    start = min(trade.opened_at for trade in trades) - timedelta(days=args.lookback_days)
    end = max(trade.closed_at for trade in trades) + timedelta(days=1)
    client = BinanceFuturesClient(timeout=args.timeout)

    candles: dict[tuple[str, str], list[Candle]] = {}
    for symbol in symbols:
        print(f"Fetching feature candles for {symbol}...")
        for interval in ("15m", "1h", "4h"):
            candles[(symbol, interval)] = completed(client.klines(symbol, interval, start, end), interval, end)

    feature_rows = [build_feature_row(trade, candles) for trade in trades]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "trade_features.csv", feature_rows)
    write_summary(output_dir / "feature_summary.csv", feature_rows)
    print_report(feature_rows, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze entry-context features for replay trades.")
    parser.add_argument("--trade-csv", action="append", required=True)
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--output-dir", default="reports/trade_feature_diagnostics")
    return parser.parse_args()


def read_trades(paths: list[Path]) -> list[TradeRow]:
    rows: list[TradeRow] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    TradeRow(
                        source=path.parent.name,
                        date=row["date"],
                        symbol=row["symbol"],
                        strategy=row["strategy"],
                        direction=row["direction"],
                        score=float(row["score"]),
                        opened_at=datetime.fromisoformat(row["opened_at"]),
                        closed_at=datetime.fromisoformat(row["closed_at"]),
                        realized_pnl=float(row["realized_pnl"]),
                        exit_reason=row["exit_reason"],
                    )
                )
    return rows


def build_feature_row(trade: TradeRow, candles: dict[tuple[str, str], list[Candle]]) -> dict[str, object]:
    symbol_15m = candles[(trade.symbol, "15m")]
    symbol_1h = candles[(trade.symbol, "1h")]
    symbol_4h = candles[(trade.symbol, "4h")]
    btc_15m = candles[("BTCUSDT", "15m")]
    btc_4h = candles[("BTCUSDT", "4h")]
    entry = trade.opened_at

    s15 = candles_until(symbol_15m, entry)
    s1h = candles_until(symbol_1h, entry)
    s4h = candles_until(symbol_4h, entry)
    b15 = candles_until(btc_15m, entry)
    b4h = candles_until(btc_4h, entry)

    last_close = s15[-1].close if s15 else 0.0
    btc_close = b15[-1].close if b15 else 0.0
    btc_ema20_4h = ema([c.close for c in b4h], 20) if len(b4h) >= 20 else 0.0
    btc_ema50_4h = ema([c.close for c in b4h], 50) if len(b4h) >= 50 else 0.0

    row = {
        "source": trade.source,
        "date": trade.date,
        "symbol": trade.symbol,
        "direction": trade.direction,
        "score": round(trade.score, 2),
        "opened_at": trade.opened_at.isoformat(),
        "hour_utc": trade.opened_at.hour + trade.opened_at.minute / 60,
        "pnl": round(trade.realized_pnl, 6),
        "is_win": trade.realized_pnl > 0,
        "exit_reason": trade.exit_reason,
        "btc_24h_return_pct": round(return_pct(b15, 96), 4),
        "btc_72h_return_pct": round(return_pct(b15, 288), 4),
        "btc_4h_slope": round(slope([c.close for c in b4h], 30), 6) if len(b4h) >= 30 else 0.0,
        "btc_4h_ema_gap_pct": round((btc_ema20_4h - btc_ema50_4h) / btc_close * 100, 4) if btc_close else 0.0,
        "symbol_24h_return_pct": round(return_pct(s15, 96), 4),
        "symbol_72h_return_pct": round(return_pct(s15, 288), 4),
        "symbol_1h_slope": round(slope([c.close for c in s1h], 30), 6) if len(s1h) >= 30 else 0.0,
        "symbol_4h_slope": round(slope([c.close for c in s4h], 30), 6) if len(s4h) >= 30 else 0.0,
        "symbol_rsi_15m": round(rsi([c.close for c in s15]), 2) if len(s15) >= 15 else 0.0,
        "symbol_atr_pct_15m": round(atr_frame_pct(s15, last_close), 4),
        "symbol_volume_expansion": round(volume_expansion(frame_like(s15)), 4) if len(s15) >= 30 else 0.0,
    }
    return row


def candles_until(candles: list[Candle], timestamp: datetime) -> list[Candle]:
    idx = index_at_or_before(candles, timestamp)
    return candles[: idx + 1] if idx is not None else []


def return_pct(candles: list[Candle], bars: int) -> float:
    if len(candles) <= bars:
        return 0.0
    start = candles[-bars - 1].close
    end = candles[-1].close
    return (end - start) / start * 100 if start else 0.0


def atr_frame_pct(candles: list[Candle], close: float) -> float:
    if len(candles) < 15 or close <= 0:
        return 0.0
    return atr(frame_like(candles)) / close * 100


def frame_like(candles: list[Candle]):
    from institutional_bot.binance_data import frame_from_candles

    return frame_from_candles(candles)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "score",
        "hour_utc",
        "btc_24h_return_pct",
        "btc_72h_return_pct",
        "btc_4h_slope",
        "btc_4h_ema_gap_pct",
        "symbol_24h_return_pct",
        "symbol_72h_return_pct",
        "symbol_1h_slope",
        "symbol_4h_slope",
        "symbol_rsi_15m",
        "symbol_atr_pct_15m",
        "symbol_volume_expansion",
    ]
    groups = {
        "winners": [row for row in rows if row["is_win"]],
        "losers": [row for row in rows if not row["is_win"]],
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "trades", "win_rate", "avg_pnl", *[f"avg_{field}" for field in fields]])
        for name, group in groups.items():
            writer.writerow([name, len(group), win_rate(group), average(group, "pnl"), *[average(group, field) for field in fields]])


def print_report(rows: list[dict[str, object]], output_dir: Path) -> None:
    winners = [row for row in rows if row["is_win"]]
    losers = [row for row in rows if not row["is_win"]]
    print()
    print("Trade feature diagnostics")
    print(f"Trades:  {len(rows)}")
    print(f"Winners: {len(winners)} avg_pnl={average(winners, 'pnl'):.4f}")
    print(f"Losers:  {len(losers)} avg_pnl={average(losers, 'pnl'):.4f}")
    for field in ("btc_24h_return_pct", "btc_72h_return_pct", "btc_4h_ema_gap_pct", "symbol_rsi_15m", "symbol_volume_expansion"):
        print(f"{field:<28} win_avg={average(winners, field):>8.4f} loss_avg={average(losers, field):>8.4f}")
    print(f"Feature CSV: {output_dir / 'trade_features.csv'}")
    print(f"Summary CSV: {output_dir / 'feature_summary.csv'}")


def average(rows: list[dict[str, object]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[field]) for row in rows) / len(rows)


def win_rate(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["is_win"]) / len(rows) * 100


if __name__ == "__main__":
    main()
