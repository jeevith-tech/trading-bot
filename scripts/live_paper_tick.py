from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from institutional_bot.binance_data import BinanceFuturesClient, BinanceMarket, Candle, frame_from_candles
from institutional_bot.config import ScannerConfig
from institutional_bot.models import MarketSnapshot, RiskState
from institutional_bot.paper import PaperConfig, PaperTradingEngine
from institutional_bot.scanner import MarketScanner
from scripts.fast_filter_optimizer import CandidateEvent, event_allowed
from scripts.paper_today_binance import (
    ReplaySettings,
    SymbolData,
    bar_imbalance,
    btc_4h_ema_gap_pct,
    btc_correlation,
    candles_at_or_before,
    completed,
    index_at_or_before,
    load_timezone,
    parse_filter_set,
    parse_strategy_set,
    prior_return_pct,
    session_liquidity,
    starting_equity_usdt,
    symbol_atr_pct,
)


TRADE_COLUMNS = [
    "date",
    "symbol",
    "strategy",
    "direction",
    "score",
    "opened_at",
    "closed_at",
    "entry",
    "exit_price",
    "quantity",
    "realized_pnl_usdt",
    "realized_pnl_inr",
    "fees_paid",
    "exit_reason",
]

DAILY_COLUMNS = [
    "date",
    "starting_equity_usdt",
    "ending_equity_usdt",
    "signals",
    "closed_trades",
    "open_positions",
    "win_rate",
    "profit_factor",
    "realized_pnl_usdt",
    "unrealized_pnl_usdt",
    "total_pnl_usdt",
    "total_pnl_inr",
    "total_pnl_pct",
    "max_drawdown_pct",
]

OPEN_POSITION_COLUMNS = [
    "date",
    "symbol",
    "strategy",
    "direction",
    "score",
    "opened_at",
    "entry",
    "stop",
    "tp1",
    "tp2",
    "quantity",
    "remaining_quantity",
    "unrealized_pnl_usdt",
]


def main() -> None:
    args = parse_args()
    tz = load_timezone(args.timezone)
    now_utc = datetime.now(timezone.utc)
    today = datetime.now(tz).date()
    state_path = Path(args.state)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path, args)
    client = BinanceFuturesClient(timeout=args.timeout)

    state = finalize_stale_day(client, state, args, tz, today)
    if state.get("active_day") != str(today):
        markets = client.top_usdt_perp_markets(limit=args.max_symbols, min_quote_volume=args.min_volume)
        if "BTCUSDT" not in {market.symbol for market in markets}:
            raise RuntimeError("BTCUSDT must be present in the live paper universe")
        state["active_day"] = str(today)
        state["day_start_equity_usdt"] = float(state["settled_equity_usdt"])
        state["active_symbols"] = [market.symbol for market in markets]

    settings = settings_for_equity(args, float(state["day_start_equity_usdt"]))
    markets = markets_for_symbols(client, list(state["active_symbols"]), args.min_volume)
    day_start = datetime.combine(today, datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)
    data = load_symbol_data(client, markets, day_start - timedelta(days=args.lookback_days), now_utc)
    result = replay_day(data, today, tz, now_utc, settings, args.inr_per_usdt, close_at_end=False)

    state["live_equity_usdt"] = result["daily_row"]["ending_equity_usdt"]
    state["last_tick_utc"] = now_utc.isoformat()
    state["last_signal_count"] = result["daily_row"]["signals"]
    state["last_closed_trades"] = result["daily_row"]["closed_trades"]
    state["last_open_positions"] = result["daily_row"]["open_positions"]

    write_json(state_path, state)
    write_reports(output_dir, state, result, args.inr_per_usdt, now_utc)
    print_status(state, result, args.inr_per_usdt, now_utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One scheduled live paper-trading tick for the v17 Binance bot.")
    parser.add_argument("--timezone", default="Asia/Calcutta")
    parser.add_argument("--state", default="reports/live_paper/state.json")
    parser.add_argument("--output-dir", default="reports/live_paper")
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--min-volume", type=float, default=50_000_000)
    parser.add_argument("--capital-inr", type=float, default=3000.0)
    parser.add_argument("--inr-per-usdt", type=float, default=95.0)
    parser.add_argument("--risk-pct", type=float, default=10.0)
    parser.add_argument("--max-risk-pct", type=float, default=10.0)
    parser.add_argument("--min-score", type=float, default=85.0)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--max-trades-per-day", type=int, default=1)
    parser.add_argument("--strategies", default="trend_continuation")
    parser.add_argument("--directions", default="long,short")
    parser.add_argument("--min-btc-21d-return-pct", type=float, default=-10.0)
    parser.add_argument("--max-btc-21d-return-pct-for-shorts", type=float, default=-10.0)
    parser.add_argument("--max-btc-72h-return-pct", type=float, default=3.0)
    parser.add_argument("--max-symbol-24h-return-pct", type=float, default=3.4)
    parser.add_argument("--min-symbol-atr-pct-15m", type=float, default=0.48)
    parser.add_argument("--skip-long-if-btc21-min", type=float, default=2.0)
    parser.add_argument("--skip-long-if-btc72-max", type=float, default=-1.0)
    parser.add_argument("--skip-long-if-symbol24-min", type=float, default=1.0)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def load_state(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "version": 1,
        "mode": "v17_free_scheduled_paper",
        "timezone": args.timezone,
        "capital_inr": args.capital_inr,
        "inr_per_usdt": args.inr_per_usdt,
        "settled_equity_usdt": starting_equity_usdt(args),
        "live_equity_usdt": starting_equity_usdt(args),
        "active_day": None,
        "day_start_equity_usdt": starting_equity_usdt(args),
        "active_symbols": [],
        "finalized_days": {},
        "finalized_trades": [],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def finalize_stale_day(
    client: BinanceFuturesClient,
    state: dict[str, Any],
    args: argparse.Namespace,
    tz,
    today: date,
) -> dict[str, Any]:
    active_day_raw = state.get("active_day")
    if not active_day_raw:
        return state
    active_day = date.fromisoformat(active_day_raw)
    if active_day >= today:
        return state

    day_start = datetime.combine(active_day, datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)
    day_end = day_start + timedelta(days=1)
    markets = markets_for_symbols(client, list(state.get("active_symbols", [])), args.min_volume)
    data = load_symbol_data(client, markets, day_start - timedelta(days=args.lookback_days), day_end)
    settings = settings_for_equity(args, float(state["day_start_equity_usdt"]))
    result = replay_day(data, active_day, tz, day_end, settings, args.inr_per_usdt, close_at_end=True)

    finalized_days = dict(state.get("finalized_days", {}))
    finalized_days[str(active_day)] = result["daily_row"]
    finalized_trades = [
        row for row in list(state.get("finalized_trades", [])) if row.get("date") != str(active_day)
    ]
    finalized_trades.extend(result["trades"])

    state["finalized_days"] = finalized_days
    state["finalized_trades"] = finalized_trades
    state["settled_equity_usdt"] = result["daily_row"]["ending_equity_usdt"]
    state["live_equity_usdt"] = result["daily_row"]["ending_equity_usdt"]
    state["active_day"] = None
    state["active_symbols"] = []
    return state


def settings_for_equity(args: argparse.Namespace, equity_usdt: float) -> ReplaySettings:
    return ReplaySettings(
        equity_usdt=equity_usdt,
        risk_pct=args.risk_pct,
        max_risk_pct=args.max_risk_pct,
        volatility_adjust_risk=False,
        min_score=args.min_score,
        min_volume=args.min_volume,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        max_positions=args.max_positions,
        flat_at_end=False,
        strategies=parse_strategy_set(args.strategies),
        directions=parse_filter_set(args.directions),
        min_btc_21d_return_pct=args.min_btc_21d_return_pct,
        max_btc_21d_return_pct_for_shorts=args.max_btc_21d_return_pct_for_shorts,
        max_btc_72h_return_pct=args.max_btc_72h_return_pct,
        max_symbol_24h_return_pct=args.max_symbol_24h_return_pct,
        min_symbol_atr_pct_15m=args.min_symbol_atr_pct_15m,
        skip_long_if_btc21_min=args.skip_long_if_btc21_min,
        skip_long_if_btc72_max=args.skip_long_if_btc72_max,
        skip_long_if_symbol24_min=args.skip_long_if_symbol24_min,
        max_trades=args.max_trades_per_day,
    )


def markets_for_symbols(client: BinanceFuturesClient, symbols: list[str], min_volume: float) -> list[BinanceMarket]:
    if not symbols:
        return client.top_usdt_perp_markets(limit=20, min_quote_volume=min_volume)
    all_markets = client.top_usdt_perp_markets(limit=500, min_quote_volume=0)
    by_symbol = {market.symbol: market for market in all_markets}
    markets = [by_symbol[symbol] for symbol in symbols if symbol in by_symbol]
    if "BTCUSDT" not in {market.symbol for market in markets}:
        btc = by_symbol.get("BTCUSDT")
        if btc is not None:
            markets.insert(0, btc)
    if not markets:
        raise RuntimeError("no frozen live paper symbols are still available on Binance")
    return markets


def load_symbol_data(
    client: BinanceFuturesClient,
    markets: list[BinanceMarket],
    start: datetime,
    end: datetime,
) -> dict[str, SymbolData]:
    data: dict[str, SymbolData] = {}
    for market in markets:
        try:
            candles_15m = completed(client.klines(market.symbol, "15m", start, end), "15m", end)
            candles_1h = completed(client.klines(market.symbol, "1h", start, end), "1h", end)
            candles_4h = completed(client.klines(market.symbol, "4h", start, end), "4h", end)
        except Exception as exc:
            print(f"Skipping {market.symbol}: {exc}")
            continue
        if len(candles_15m) < 80 or len(candles_1h) < 40 or len(candles_4h) < 20:
            continue
        data[market.symbol] = SymbolData(market, candles_15m, candles_1h, candles_4h, 0.0)
    if "BTCUSDT" not in data:
        raise RuntimeError("BTCUSDT data is required for live paper scoring")
    return data


def replay_day(
    data: dict[str, SymbolData],
    day: date,
    tz,
    replay_end_utc: datetime,
    settings: ReplaySettings,
    inr_per_usdt: float,
    close_at_end: bool,
) -> dict[str, Any]:
    day_start = datetime.combine(day, datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)
    day_end = day_start + timedelta(days=1)
    replay_end = min(replay_end_utc, day_end)
    timestamps = [
        candle.open_time
        for candle in data["BTCUSDT"].candles_15m
        if day_start <= candle.open_time < replay_end
    ]
    if not timestamps:
        return empty_result(day, settings.equity_usdt)

    engine = PaperTradingEngine(
        PaperConfig(
            starting_equity=settings.equity_usdt,
            fee_bps=settings.fee_bps,
            slippage_bps=settings.slippage_bps,
            max_concurrent_positions=settings.max_positions,
        )
    )
    scanner = MarketScanner(
        ScannerConfig(
            min_daily_volume_usd=settings.min_volume,
            min_score_to_trade=settings.min_score,
            risk_per_trade_pct=settings.risk_pct,
            max_risk_per_trade_pct=settings.max_risk_pct,
            volatility_adjust_risk=False,
        )
    )
    neutral_risk = RiskState(equity=settings.equity_usdt)
    signal_count = 0
    last_marks: dict[str, float] = {}

    for timestamp in timestamps:
        for position in [position for position in engine.positions if position.is_open]:
            candle = candle_at(data.get(position.candidate.symbol), timestamp)
            if candle is not None:
                last_marks[position.candidate.symbol] = candle.close
                engine.update_bar(position.candidate.symbol, timestamp, candle.high, candle.low, candle.close)

        snapshots: list[MarketSnapshot] = []
        for symbol, symbol_data in data.items():
            index_15m = index_at_or_before(symbol_data.candles_15m, timestamp)
            if index_15m is None:
                continue
            current = symbol_data.candles_15m[index_15m]
            last_marks[symbol] = current.close
            if index_15m < 80:
                continue
            candles_15m = symbol_data.candles_15m[: index_15m + 1]
            candles_1h = candles_at_or_before(symbol_data.candles_1h, timestamp)
            candles_4h = candles_at_or_before(symbol_data.candles_4h, timestamp)
            if len(candles_1h) < 40 or len(candles_4h) < 20:
                continue
            snapshots.append(
                MarketSnapshot(
                    symbol=symbol,
                    price=current.close,
                    daily_volume_usd=symbol_data.market.quote_volume,
                    spread_bps=symbol_data.market.spread_bps,
                    funding_rate=symbol_data.market.funding_rate,
                    open_interest_change_pct=0.0,
                    btc_correlation=btc_correlation(symbol, candles_15m, data["BTCUSDT"].candles_15m),
                    order_book_imbalance=bar_imbalance(current),
                    session_liquidity_score=session_liquidity(symbol_data.market.quote_volume),
                    exchange_health_score=1.0,
                    news_risk_score=0.0,
                    candles={
                        "15m": frame_from_candles(candles_15m),
                        "1h": frame_from_candles(candles_1h),
                        "4h": frame_from_candles(candles_4h),
                    },
                )
            )

        if settings.max_trades is not None and signal_count >= settings.max_trades:
            continue
        if any(position.is_open for position in engine.positions):
            continue

        decision = scanner.scan(snapshots, neutral_risk)
        for candidate, plan in decision.tradable:
            symbol_data = data[candidate.symbol]
            event = CandidateEvent(
                day=day,
                day_start=day_start,
                day_end=day_end,
                timestamp=timestamp,
                candidate=candidate,
                btc_24h_return_pct=prior_return_pct(data["BTCUSDT"].candles_15m, timestamp, 96),
                btc_72h_return_pct=prior_return_pct(data["BTCUSDT"].candles_15m, timestamp, 288),
                btc_4h_ema_gap_pct=btc_4h_ema_gap_pct(data["BTCUSDT"].candles_4h, timestamp),
                symbol_24h_return_pct=prior_return_pct(symbol_data.candles_15m, timestamp, 96),
                btc_21d_return_pct=prior_return_pct(data["BTCUSDT"].candles_15m, timestamp, 96 * 21),
                symbol_atr_pct_15m=symbol_atr_pct(symbol_data.candles_15m, timestamp),
            )
            if not event_allowed(event, settings):
                continue
            if engine.can_open(candidate.symbol):
                engine.open_position(candidate, plan, timestamp)
                signal_count += 1
                break

    if close_at_end:
        engine.close_all(timestamps[-1], last_marks, reason="session_end")

    summary = engine.summary(last_marks)
    daily_row = {
        "date": str(day),
        "starting_equity_usdt": summary.starting_equity,
        "ending_equity_usdt": summary.equity,
        "signals": signal_count,
        "closed_trades": summary.closed_trades,
        "open_positions": summary.open_positions,
        "win_rate": summary.win_rate,
        "profit_factor": summary.profit_factor,
        "realized_pnl_usdt": summary.realized_pnl,
        "unrealized_pnl_usdt": summary.unrealized_pnl,
        "total_pnl_usdt": summary.total_pnl,
        "total_pnl_inr": summary.total_pnl * inr_per_usdt,
        "total_pnl_pct": summary.total_pnl_pct,
        "max_drawdown_pct": summary.max_drawdown_pct,
    }
    return {
        "daily_row": daily_row,
        "trades": trade_rows(engine, day, inr_per_usdt),
        "open_positions": open_position_rows(engine, day, last_marks),
    }


def candle_at(symbol_data: SymbolData | None, timestamp: datetime) -> Candle | None:
    if symbol_data is None:
        return None
    idx = index_at_or_before(symbol_data.candles_15m, timestamp)
    if idx is None or symbol_data.candles_15m[idx].open_time != timestamp:
        return None
    return symbol_data.candles_15m[idx]


def empty_result(day: date, starting_equity: float) -> dict[str, Any]:
    return {
        "daily_row": {
            "date": str(day),
            "starting_equity_usdt": starting_equity,
            "ending_equity_usdt": starting_equity,
            "signals": 0,
            "closed_trades": 0,
            "open_positions": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "realized_pnl_usdt": 0.0,
            "unrealized_pnl_usdt": 0.0,
            "total_pnl_usdt": 0.0,
            "total_pnl_inr": 0.0,
            "total_pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
        },
        "trades": [],
        "open_positions": [],
    }


def trade_rows(engine: PaperTradingEngine, day: date, inr_per_usdt: float) -> list[dict[str, Any]]:
    rows = []
    for trade in engine.trades:
        rows.append(
            {
                "date": str(day),
                "symbol": trade.symbol,
                "strategy": trade.strategy,
                "direction": trade.direction,
                "score": trade.score,
                "opened_at": trade.opened_at.isoformat(),
                "closed_at": trade.closed_at.isoformat(),
                "entry": trade.entry,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "realized_pnl_usdt": trade.realized_pnl,
                "realized_pnl_inr": trade.realized_pnl * inr_per_usdt,
                "fees_paid": trade.fees_paid,
                "exit_reason": trade.exit_reason,
            }
        )
    return rows


def open_position_rows(engine: PaperTradingEngine, day: date, marks: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for position in engine.positions:
        if not position.is_open:
            continue
        rows.append(
            {
                "date": str(day),
                "symbol": position.candidate.symbol,
                "strategy": position.candidate.strategy.value,
                "direction": position.candidate.direction.value,
                "score": position.candidate.score,
                "opened_at": position.opened_at.isoformat(),
                "entry": position.entry,
                "stop": position.stop,
                "tp1": position.tp1,
                "tp2": position.tp2,
                "quantity": position.quantity,
                "remaining_quantity": position.remaining_quantity,
                "unrealized_pnl_usdt": engine._unrealized(position, marks),
            }
        )
    return rows


def write_reports(
    output_dir: Path,
    state: dict[str, Any],
    current_result: dict[str, Any],
    inr_per_usdt: float,
    now_utc: datetime,
) -> None:
    finalized_days = dict(state.get("finalized_days", {}))
    active_day = str(current_result["daily_row"]["date"])
    daily_rows = [finalized_days[key] for key in sorted(finalized_days)]
    if active_day not in finalized_days:
        daily_rows.append(current_result["daily_row"])

    trades = list(state.get("finalized_trades", []))
    if active_day not in finalized_days:
        trades.extend(current_result["trades"])

    write_csv(output_dir / "daily_summary.csv", DAILY_COLUMNS, daily_rows)
    write_csv(output_dir / "trades.csv", TRADE_COLUMNS, trades)
    write_csv(output_dir / "open_positions.csv", OPEN_POSITION_COLUMNS, current_result["open_positions"])
    write_status_markdown(output_dir / "status.md", state, current_result, inr_per_usdt, now_utc)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_cell(row.get(field, "")) for field in fieldnames})


def format_cell(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.8f}"
    return value


def write_status_markdown(
    path: Path,
    state: dict[str, Any],
    current_result: dict[str, Any],
    inr_per_usdt: float,
    now_utc: datetime,
) -> None:
    row = current_result["daily_row"]
    settled_inr = float(state["settled_equity_usdt"]) * inr_per_usdt
    live_inr = float(row["ending_equity_usdt"]) * inr_per_usdt
    lines = [
        "# Live Paper Trading Status",
        "",
        f"- Updated UTC: `{now_utc.isoformat()}`",
        f"- Active day: `{row['date']}`",
        f"- Mode: `{state.get('mode', 'v17_free_scheduled_paper')}`",
        f"- Settled equity: `{settled_inr:.2f} INR`",
        f"- Live equity: `{live_inr:.2f} INR`",
        f"- Today's PnL: `{float(row['total_pnl_inr']):.2f} INR`",
        f"- Signals today: `{row['signals']}`",
        f"- Closed trades today: `{row['closed_trades']}`",
        f"- Open positions: `{row['open_positions']}`",
        f"- Win rate today: `{float(row['win_rate']):.2f}%`",
        "",
    ]
    if current_result["open_positions"]:
        lines.append("## Open Positions")
        lines.append("")
        for position in current_result["open_positions"]:
            lines.append(
                f"- `{position['symbol']}` `{position['direction']}` score `{float(position['score']):.2f}` "
                f"unrealized `{float(position['unrealized_pnl_usdt']) * inr_per_usdt:.2f} INR`"
            )
    else:
        lines.append("No open paper positions right now.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def print_status(state: dict[str, Any], result: dict[str, Any], inr_per_usdt: float, now_utc: datetime) -> None:
    row = result["daily_row"]
    print("Live paper tick complete")
    print(f"Updated UTC:      {now_utc.isoformat()}")
    print(f"Active day:       {row['date']}")
    print(f"Live equity:      {float(row['ending_equity_usdt']) * inr_per_usdt:.2f} INR")
    print(f"Today's PnL:      {float(row['total_pnl_inr']):.2f} INR")
    print(f"Signals today:    {row['signals']}")
    print(f"Closed trades:    {row['closed_trades']}")
    print(f"Open positions:   {row['open_positions']}")


if __name__ == "__main__":
    main()
