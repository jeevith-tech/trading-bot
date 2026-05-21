from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from institutional_bot.binance_data import BinanceFuturesClient
from scripts.paper_today_binance import (
    ReplaySettings,
    ReplayResult,
    SymbolData,
    completed,
    load_timezone,
    parse_strategy_set,
    parse_filter_set,
    run_replay,
    starting_equity_usdt,
    strategy_label,
    write_trades,
)


def main() -> None:
    args = parse_args()
    tz = load_timezone(args.timezone)
    days = selected_days(args, tz)
    if not days:
        raise RuntimeError("no test days selected")

    first_start_local = datetime.combine(days[0], datetime.min.time(), tzinfo=tz)
    last_end_local = datetime.combine(days[-1] + timedelta(days=1), datetime.min.time(), tzinfo=tz)
    fetch_start = first_start_local.astimezone(timezone.utc) - timedelta(days=args.lookback_days)
    fetch_end = last_end_local.astimezone(timezone.utc)

    client = BinanceFuturesClient(timeout=args.timeout)
    print(
        f"Fetching Binance data for {len(days)} completed days "
        f"({days[0]} to {days[-1]}, {args.timezone})..."
    )
    markets = client.top_usdt_perp_markets(limit=args.max_symbols, min_quote_volume=args.min_volume)
    print(f"Selected {len(markets)} liquid crypto markets. Top symbols: {', '.join(market.symbol for market in markets[:10])}")

    base_data: dict[str, SymbolData] = {}
    for offset, market in enumerate(markets, start=1):
        try:
            candles_15m = completed(client.klines(market.symbol, "15m", fetch_start, fetch_end), "15m", fetch_end)
            candles_1h = completed(client.klines(market.symbol, "1h", fetch_start, fetch_end), "1h", fetch_end)
            candles_4h = completed(client.klines(market.symbol, "4h", fetch_start, fetch_end), "4h", fetch_end)
        except Exception as exc:
            print(f"Skipping {market.symbol}: {exc}")
            continue
        if len(candles_15m) < 80 or len(candles_1h) < 40 or len(candles_4h) < 20:
            continue
        base_data[market.symbol] = SymbolData(market, candles_15m, candles_1h, candles_4h, 0.0)
        if offset % 20 == 0:
            print(f"Fetched candles for {offset}/{len(markets)} markets...")

    if "BTCUSDT" not in base_data:
        raise RuntimeError("BTCUSDT data is required for correlation scoring")

    max_trades = args.max_trades_per_day
    if args.one_trade_per_day or args.optimize_one_trade:
        max_trades = 1

    base_settings = ReplaySettings(
        equity_usdt=starting_equity_usdt(args),
        risk_pct=args.risk_pct,
        max_risk_pct=args.max_risk_pct,
        volatility_adjust_risk=not args.fixed_risk,
        min_score=args.min_score,
        min_volume=args.min_volume,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        max_positions=args.max_positions,
        flat_at_end=True,
        strategies=parse_strategy_set(args.strategies),
        directions=parse_filter_set(args.directions),
        min_minutes_before_close=args.min_minutes_before_close,
        min_minutes_after_open=args.min_minutes_after_open,
        min_btc_24h_return_pct=args.min_btc_24h_return_pct,
        min_btc_21d_return_pct=args.min_btc_21d_return_pct,
        max_btc_24h_return_pct_for_shorts=args.max_btc_24h_return_pct_for_shorts,
        max_btc_21d_return_pct_for_shorts=args.max_btc_21d_return_pct_for_shorts,
        max_btc_72h_return_pct=args.max_btc_72h_return_pct,
        max_btc_72h_return_pct_for_shorts=args.max_btc_72h_return_pct_for_shorts,
        max_btc_4h_ema_gap_pct=args.max_btc_4h_ema_gap_pct,
        max_symbol_24h_return_pct=args.max_symbol_24h_return_pct,
        min_symbol_atr_pct_15m=args.min_symbol_atr_pct_15m,
        breakeven_trigger_r=args.breakeven_trigger_r,
        soft_stop_r=args.soft_stop_r,
        max_trades=max_trades,
    )

    day_data_by_date = {}
    for day in days:
        day_start = datetime.combine(day, datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)
        day_end = day_start + timedelta(days=1)
        print(f"Preparing {day} open-interest context...")
        if args.skip_open_interest or args.optimize_one_trade:
            day_data_by_date[day] = base_data
        else:
            day_data_by_date[day] = with_day_open_interest(client, base_data, day_start, day_end)

    if args.optimize_one_trade:
        ranked = optimize_one_trade_settings(day_data_by_date, days, tz, base_settings, args.min_trade_days)
        print_optimizer_table(ranked)
        base_settings = ranked[0][0]
        print()
        print(
            "Best one-trade daily filter selected: "
            f"min_score={base_settings.min_score:.1f}, "
            f"strategies={strategy_label(base_settings.strategies)}"
        )

    report_rows, all_trade_rows, replays = replay_days(
        day_data_by_date,
        days,
        tz,
        base_settings,
        args.inr_per_usdt,
        verbose=True,
    )

    output_root = Path(args.output_dir)
    for day, replay in replays.items():
        day_trade_path = output_root / f"binance_paper_{day}_trades.csv"
        day_trade_path.parent.mkdir(parents=True, exist_ok=True)
        write_trades(day_trade_path, replay.engine)

    summary_path = output_root / "binance_multi_day_summary.csv"
    trades_path = output_root / "binance_multi_day_trades.csv"
    write_summary(summary_path, report_rows)
    write_all_trades(trades_path, all_trade_rows)
    print_report(report_rows, base_settings, args.inr_per_usdt, summary_path, trades_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Binance paper trading over multiple completed days.")
    parser.add_argument("--timezone", default="Asia/Calcutta")
    parser.add_argument("--days", type=int, default=5, help="Number of completed local days to test.")
    parser.add_argument("--end-date", default="", help="Last local date to test, YYYY-MM-DD. Defaults to yesterday.")
    parser.add_argument("--max-symbols", type=int, default=140)
    parser.add_argument("--min-volume", type=float, default=50_000_000)
    parser.add_argument("--equity", type=float, default=100_000)
    parser.add_argument("--capital-inr", type=float, default=None)
    parser.add_argument("--inr-per-usdt", type=float, default=95.0)
    parser.add_argument("--risk-pct", type=float, default=10.0)
    parser.add_argument("--max-risk-pct", type=float, default=10.0)
    parser.add_argument("--fixed-risk", action="store_true", help="Use exactly --risk-pct per trade instead of volatility-reducing risk.")
    parser.add_argument("--min-score", type=float, default=89.0)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--max-trades-per-day", type=int, default=None)
    parser.add_argument("--one-trade-per-day", action="store_true")
    parser.add_argument("--optimize-one-trade", action="store_true")
    parser.add_argument("--min-trade-days", type=int, default=5)
    parser.add_argument("--skip-open-interest", action="store_true")
    parser.add_argument("--strategies", default="")
    parser.add_argument("--directions", default="")
    parser.add_argument("--min-minutes-before-close", type=int, default=0)
    parser.add_argument("--min-minutes-after-open", type=int, default=0)
    parser.add_argument("--min-btc-24h-return-pct", type=float, default=None)
    parser.add_argument("--min-btc-21d-return-pct", type=float, default=None)
    parser.add_argument("--max-btc-24h-return-pct-for-shorts", type=float, default=None)
    parser.add_argument("--max-btc-21d-return-pct-for-shorts", type=float, default=None)
    parser.add_argument("--max-btc-72h-return-pct", type=float, default=None)
    parser.add_argument("--max-btc-72h-return-pct-for-shorts", type=float, default=None)
    parser.add_argument("--max-btc-4h-ema-gap-pct", type=float, default=None)
    parser.add_argument("--max-symbol-24h-return-pct", type=float, default=None)
    parser.add_argument("--min-symbol-atr-pct-15m", type=float, default=None)
    parser.add_argument("--breakeven-trigger-r", type=float, default=None)
    parser.add_argument("--soft-stop-r", type=float, default=None)
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--output-dir", default="reports/multi_day")
    return parser.parse_args()


def selected_days(args: argparse.Namespace, tz) -> list[date]:
    if args.end_date:
        end = date.fromisoformat(args.end_date)
    else:
        end = datetime.now(tz).date() - timedelta(days=1)
    start = end - timedelta(days=args.days - 1)
    return [start + timedelta(days=offset) for offset in range(args.days)]


def with_day_open_interest(
    client: BinanceFuturesClient,
    base_data: dict[str, SymbolData],
    day_start: datetime,
    day_end: datetime,
) -> dict[str, SymbolData]:
    day_data: dict[str, SymbolData] = {}
    for symbol, data in base_data.items():
        oi_change = client.open_interest_change_pct(symbol, day_start, day_end, period="1h")
        day_data[symbol] = replace(data, oi_change_pct=oi_change)
    return day_data


def replay_days(
    day_data_by_date: dict[date, dict[str, SymbolData]],
    days: list[date],
    tz,
    base_settings: ReplaySettings,
    inr_per_usdt: float,
    verbose: bool = False,
) -> tuple[list[dict[str, object]], list[list[str]], dict[date, ReplayResult]]:
    report_rows = []
    all_trade_rows = []
    replays = {}
    current_equity_usdt = base_settings.equity_usdt
    for day in days:
        day_start = datetime.combine(day, datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)
        day_end = day_start + timedelta(days=1)
        if verbose:
            print(f"Replaying {day}...")
        settings = replace(base_settings, equity_usdt=current_equity_usdt)
        replay = run_replay(day_data_by_date[day], day_start, day_end, settings)
        replays[day] = replay
        summary = replay.summary
        current_equity_usdt = summary.equity
        report_rows.append(
            {
                "date": str(day),
                "starting_equity_usdt": summary.starting_equity,
                "ending_equity_usdt": summary.equity,
                "signals": replay.signal_count,
                "closed_trades": summary.closed_trades,
                "win_rate": summary.win_rate,
                "profit_factor": summary.profit_factor,
                "pnl_usdt": summary.total_pnl,
                "pnl_inr": summary.total_pnl * inr_per_usdt,
                "pnl_pct": summary.total_pnl_pct,
                "max_drawdown_pct": summary.max_drawdown_pct,
            }
        )
        for trade in replay.engine.trades:
            all_trade_rows.append(
                [
                    str(day),
                    trade.symbol,
                    trade.strategy,
                    trade.direction,
                    f"{trade.score:.2f}",
                    trade.opened_at.isoformat(),
                    trade.closed_at.isoformat(),
                    f"{trade.realized_pnl:.6f}",
                    trade.exit_reason,
                ]
            )
    return report_rows, all_trade_rows, replays


def optimize_one_trade_settings(
    day_data_by_date: dict[date, dict[str, SymbolData]],
    days: list[date],
    tz,
    base_settings: ReplaySettings,
    min_trade_days: int,
) -> list[tuple[ReplaySettings, list[dict[str, object]]]]:
    strategy_sets = (
        frozenset(),
        frozenset({"momentum_breakout"}),
        frozenset({"trend_continuation"}),
        frozenset({"liquidity_sweep"}),
        frozenset({"momentum_breakout", "liquidity_sweep"}),
    )
    min_scores = (90.0, 91.0, 92.0, 93.0, 94.0, 95.0)
    ranked: list[tuple[ReplaySettings, list[dict[str, object]]]] = []
    for strategies in strategy_sets:
        for min_score in min_scores:
            settings = replace(
                base_settings,
                min_score=min_score,
                max_positions=1,
                max_trades=1,
                strategies=strategies,
            )
            rows, _, _ = replay_days(day_data_by_date, days, tz, settings, inr_per_usdt=1.0)
            if trade_day_count(rows) < min_trade_days:
                continue
            ranked.append((settings, rows))
    ranked.sort(
        key=lambda item: (
            aggregate_win_rate(item[1]),
            aggregate_pnl(item[1]),
            trade_day_count(item[1]),
        ),
        reverse=True,
    )
    if not ranked:
        raise RuntimeError(f"no one-trade settings produced trades on at least {min_trade_days} days")
    return ranked[:10]


def aggregate_win_rate(rows: list[dict[str, object]]) -> float:
    total_trades = sum(int(row["closed_trades"]) for row in rows)
    if total_trades == 0:
        return 0.0
    wins = sum(float(row["win_rate"]) * int(row["closed_trades"]) / 100 for row in rows)
    return wins / total_trades * 100


def aggregate_pnl(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return float(rows[-1]["ending_equity_usdt"]) - float(rows[0]["starting_equity_usdt"])


def trade_day_count(rows: list[dict[str, object]]) -> int:
    return sum(1 for row in rows if int(row["closed_trades"]) > 0)


def print_optimizer_table(ranked: list[tuple[ReplaySettings, list[dict[str, object]]]]) -> None:
    print()
    print("Top one-trade-per-day candidates:")
    for idx, (settings, rows) in enumerate(ranked, start=1):
        total_trades = sum(int(row["closed_trades"]) for row in rows)
        print(
            f"{idx:>2}. win={aggregate_win_rate(rows):>5.1f}% "
            f"trade_days={trade_day_count(rows):>2} "
            f"trades={total_trades:>2} "
            f"pnl={aggregate_pnl(rows):>7.2f} USDT "
            f"score>={settings.min_score:>4.1f} "
            f"strategies={strategy_label(settings.strategies)}"
        )


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "starting_equity_usdt",
                "ending_equity_usdt",
                "signals",
                "closed_trades",
                "win_rate",
                "profit_factor",
                "pnl_usdt",
                "pnl_inr",
                "pnl_pct",
                "max_drawdown_pct",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_all_trades(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "symbol", "strategy", "direction", "score", "opened_at", "closed_at", "realized_pnl", "exit_reason"])
        writer.writerows(rows)


def print_report(
    rows: list[dict[str, object]],
    settings: ReplaySettings,
    inr_per_usdt: float,
    summary_path: Path,
    trades_path: Path,
) -> None:
    starting_equity = float(rows[0]["starting_equity_usdt"]) if rows else settings.equity_usdt
    ending_equity = float(rows[-1]["ending_equity_usdt"]) if rows else settings.equity_usdt
    total_usdt = ending_equity - starting_equity
    total_inr = total_usdt * inr_per_usdt
    total_trades = sum(int(row["closed_trades"]) for row in rows)
    wins_weighted = sum(float(row["win_rate"]) * int(row["closed_trades"]) / 100 for row in rows)
    win_rate = wins_weighted / total_trades * 100 if total_trades else 0.0
    print()
    print("Multi-day paper PnL")
    print(
        f"Settings: capital={settings.equity_usdt:.2f} USDT, "
        f"min_score={settings.min_score:.1f}, max_positions={settings.max_positions}, "
        f"strategies={strategy_label(settings.strategies)}"
    )
    print("date         start_usdt  end_usdt  trades  win%    pnl_usdt   pnl_inr   dd%")
    for row in rows:
        print(
            f"{row['date']}  "
            f"{float(row['starting_equity_usdt']):>10.2f}  "
            f"{float(row['ending_equity_usdt']):>8.2f}  "
            f"{int(row['closed_trades']):>6}  "
            f"{float(row['win_rate']):>5.1f}  "
            f"{float(row['pnl_usdt']):>10.2f}  "
            f"{float(row['pnl_inr']):>8.2f}  "
            f"{float(row['max_drawdown_pct']):>5.2f}"
        )
    print(f"Total trades:          {total_trades}")
    print(f"Weighted win rate:     {win_rate:.1f}%")
    print(f"Starting equity:       {starting_equity:.2f} USDT")
    print(f"Ending equity:         {ending_equity:.2f} USDT")
    print(f"Total PnL:             {total_usdt:.2f} USDT")
    print(f"Estimated INR PnL:     {total_inr:.2f} INR")
    print(f"Summary CSV:           {summary_path}")
    print(f"All trades CSV:        {trades_path}")


if __name__ == "__main__":
    main()
