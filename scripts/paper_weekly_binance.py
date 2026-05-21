from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from institutional_bot.binance_data import BinanceFuturesClient, OpenInterestPoint
from scripts.paper_multi_day_binance import replay_days, write_all_trades, write_summary
from scripts.paper_today_binance import (
    ReplaySettings,
    SymbolData,
    completed,
    load_timezone,
    parse_filter_set,
    parse_strategy_set,
    starting_equity_usdt,
    strategy_label,
)


@dataclass(frozen=True)
class VariantResult:
    name: str
    settings: ReplaySettings
    rows: list[dict[str, object]]
    trades: list[list[str]]
    weekly_rows: list[dict[str, object]]


def main() -> None:
    args = parse_args()
    tz = load_timezone(args.timezone)
    days = selected_days(args, tz)
    first_start = datetime.combine(days[0], datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)
    final_end = datetime.combine(days[-1] + timedelta(days=1), datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)
    fetch_start = first_start - timedelta(days=args.lookback_days)

    client = BinanceFuturesClient(timeout=args.timeout)
    print(f"Fetching Binance data for {len(days)} days ({days[0]} to {days[-1]}, {args.timezone})...")
    markets = client.top_usdt_perp_markets(
        limit=max(args.max_symbols, args.candidate_symbols),
        min_quote_volume=args.min_volume,
    )
    print(f"Selected {len(markets)} liquid crypto markets. Top symbols: {', '.join(market.symbol for market in markets[:10])}")

    base_data: dict[str, SymbolData] = {}
    oi_history: dict[str, list[OpenInterestPoint]] = {}
    for offset, market in enumerate(markets, start=1):
        try:
            candles_15m = completed(client.klines(market.symbol, "15m", fetch_start, final_end), "15m", final_end)
            candles_1h = completed(client.klines(market.symbol, "1h", fetch_start, final_end), "1h", final_end)
            candles_4h = completed(client.klines(market.symbol, "4h", fetch_start, final_end), "4h", final_end)
        except Exception as exc:
            print(f"Skipping {market.symbol}: {exc}")
            continue
        if args.skip_open_interest:
            oi_history[market.symbol] = []
        else:
            try:
                oi_history[market.symbol] = client.open_interest_history(
                    market.symbol,
                    first_start,
                    final_end,
                    period="1h",
                )
            except Exception as exc:
                print(f"Open interest unavailable for {market.symbol}: {exc}")
                oi_history[market.symbol] = []
        if len(candles_15m) < 80 or len(candles_1h) < 40 or len(candles_4h) < 20:
            continue
        base_data[market.symbol] = SymbolData(market, candles_15m, candles_1h, candles_4h, 0.0)
        if offset % 10 == 0:
            print(f"Fetched {offset}/{len(markets)} markets...")

    if "BTCUSDT" not in base_data:
        raise RuntimeError("BTCUSDT data is required for correlation scoring")

    if args.historical_volume_ranking:
        base_data = rank_by_historical_volume(base_data, first_start, final_end, args.max_symbols)
        oi_history = {symbol: oi_history.get(symbol, []) for symbol in base_data}
        print(f"Historical-volume universe: {', '.join(base_data.keys())}")

    print("Preparing day-level open-interest context...")
    day_data_by_date = {
        day: apply_day_open_interest(
            base_data,
            oi_history,
            datetime.combine(day, datetime.min.time(), tzinfo=tz).astimezone(timezone.utc),
            datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=tz).astimezone(timezone.utc),
        )
        for day in days
    }

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
        max_trades=args.max_trades_per_day,
    )

    variants = build_variants(args, base_settings)
    results: list[VariantResult] = []
    for idx, (name, settings) in enumerate(variants, start=1):
        print(f"Testing variant {idx}/{len(variants)}: {name}")
        rows, trades, _ = replay_days(day_data_by_date, days, tz, settings, args.inr_per_usdt)
        results.append(
            VariantResult(
                name=name,
                settings=settings,
                rows=rows,
                trades=trades,
                weekly_rows=weekly_rollup(rows, (len(days) + 6) // 7),
            )
        )

    ranked = sorted(results, key=variant_rank_key, reverse=True)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    write_variant_summary(output_root / "weekly_variant_summary.csv", ranked, args.inr_per_usdt)
    best = ranked[0]
    write_summary(output_root / "best_daily_summary.csv", best.rows)
    write_weekly_rows(output_root / "best_weekly_summary.csv", best.weekly_rows)
    write_all_trades(output_root / "best_trades.csv", best.trades)
    print_report(ranked, output_root, args.inr_per_usdt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paper trading variants across separate Binance weeks.")
    parser.add_argument("--timezone", default="Asia/Calcutta")
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--start-date", default="", help="First local date to test, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="", help="Last local date to test, YYYY-MM-DD. Defaults to yesterday.")
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--candidate-symbols", type=int, default=20)
    parser.add_argument("--historical-volume-ranking", action="store_true")
    parser.add_argument("--min-volume", type=float, default=50_000_000)
    parser.add_argument("--equity", type=float, default=100_000)
    parser.add_argument("--capital-inr", type=float, default=None)
    parser.add_argument("--inr-per-usdt", type=float, default=95.0)
    parser.add_argument("--risk-pct", type=float, default=10.0)
    parser.add_argument("--max-risk-pct", type=float, default=10.0)
    parser.add_argument("--fixed-risk", action="store_true", help="Use exactly --risk-pct per trade instead of volatility-reducing risk.")
    parser.add_argument("--min-score", type=float, default=94.0)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--max-trades-per-day", type=int, default=1)
    parser.add_argument("--strategies", default="momentum_breakout")
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
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--thresholds", default="90,92,94,95")
    parser.add_argument("--strategy-sets", default="momentum_breakout;momentum_breakout,liquidity_sweep;all")
    parser.add_argument("--direction-sets", default="all")
    parser.add_argument("--close-buffer-sets", default="0")
    parser.add_argument("--btc-return-sets", default="")
    parser.add_argument("--after-open-sets", default="")
    parser.add_argument("--btc-72h-max-sets", default="")
    parser.add_argument("--btc-4h-ema-gap-max-sets", default="")
    parser.add_argument("--symbol-24h-max-sets", default="")
    parser.add_argument("--symbol-atr-min-sets", default="")
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--skip-open-interest", action="store_true")
    parser.add_argument("--output-dir", default="reports/weekly_validation")
    return parser.parse_args()


def selected_days(args: argparse.Namespace, tz) -> list[date]:
    if args.end_date:
        end = date.fromisoformat(args.end_date)
    else:
        end = datetime.now(tz).date() - timedelta(days=1)
    if args.start_date:
        start = date.fromisoformat(args.start_date)
    else:
        start = end - timedelta(days=args.weeks * 7 - 1)
    if start > end:
        raise ValueError("--start-date must be before or equal to --end-date")
    day_count = (end - start).days + 1
    return [start + timedelta(days=offset) for offset in range(day_count)]


def apply_day_open_interest(
    base_data: dict[str, SymbolData],
    oi_history: dict[str, list[OpenInterestPoint]],
    day_start: datetime,
    day_end: datetime,
) -> dict[str, SymbolData]:
    result: dict[str, SymbolData] = {}
    for symbol, data in base_data.items():
        result[symbol] = replace(
            data,
            oi_change_pct=open_interest_change_for_window(oi_history.get(symbol, []), day_start, day_end),
        )
    return result


def rank_by_historical_volume(
    base_data: dict[str, SymbolData],
    start: datetime,
    end: datetime,
    limit: int,
) -> dict[str, SymbolData]:
    ranked = sorted(
        base_data.items(),
        key=lambda item: sum(candle.quote_volume for candle in item[1].candles_15m if start <= candle.open_time < end),
        reverse=True,
    )
    selected = dict(ranked[:limit])
    if "BTCUSDT" not in selected and "BTCUSDT" in base_data:
        if len(selected) >= limit:
            selected.pop(next(reversed(selected)))
        selected["BTCUSDT"] = base_data["BTCUSDT"]
    return selected


def open_interest_change_for_window(points: list[OpenInterestPoint], start: datetime, end: datetime) -> float:
    window = [point for point in points if start <= point.timestamp < end and point.open_interest > 0]
    if len(window) < 2:
        return 0.0
    first = window[0].open_interest
    last = window[-1].open_interest
    return (last - first) / first * 100 if first else 0.0


def build_variants(args: argparse.Namespace, base: ReplaySettings) -> list[tuple[str, ReplaySettings]]:
    if not args.compare:
        return [(variant_name(base), base)]
    variants: list[tuple[str, ReplaySettings]] = []
    thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    strategy_sets = [value.strip() for value in args.strategy_sets.split(";") if value.strip()]
    direction_sets = [value.strip() for value in args.direction_sets.split(";") if value.strip()]
    close_buffers = [int(value.strip()) for value in args.close_buffer_sets.split(",") if value.strip()]
    btc_return_sets = float_sets(args.btc_return_sets, base.min_btc_24h_return_pct)
    after_open_sets = int_sets(args.after_open_sets, base.min_minutes_after_open)
    btc_72h_max_sets = float_sets(args.btc_72h_max_sets, base.max_btc_72h_return_pct)
    btc_4h_gap_sets = float_sets(args.btc_4h_ema_gap_max_sets, base.max_btc_4h_ema_gap_pct)
    symbol_24h_max_sets = float_sets(args.symbol_24h_max_sets, base.max_symbol_24h_return_pct)
    symbol_atr_min_sets = float_sets(args.symbol_atr_min_sets, base.min_symbol_atr_pct_15m)
    for symbol_atr_min in symbol_atr_min_sets:
        for symbol_24h_max in symbol_24h_max_sets:
            for btc_4h_gap in btc_4h_gap_sets:
                for btc_72h_max in btc_72h_max_sets:
                    for after_open in after_open_sets:
                        for btc_return in btc_return_sets:
                            for close_buffer in close_buffers:
                                for raw_directions in direction_sets:
                                    directions = frozenset() if raw_directions == "all" else parse_filter_set(raw_directions)
                                    for raw_strategies in strategy_sets:
                                        strategies = frozenset() if raw_strategies == "all" else parse_strategy_set(raw_strategies)
                                        for threshold in thresholds:
                                            settings = replace(
                                                base,
                                                min_score=threshold,
                                                strategies=strategies,
                                                directions=directions,
                                                min_minutes_before_close=close_buffer,
                                                min_minutes_after_open=after_open,
                                                min_btc_24h_return_pct=btc_return,
                                                max_btc_72h_return_pct=btc_72h_max,
                                                max_btc_4h_ema_gap_pct=btc_4h_gap,
                                                max_symbol_24h_return_pct=symbol_24h_max,
                                                min_symbol_atr_pct_15m=symbol_atr_min,
                                            )
                                            variants.append((variant_name(settings), settings))
    return variants


def float_sets(raw: str, base_value: float | None) -> list[float | None]:
    if not raw:
        return [base_value]
    values: list[float | None] = []
    for value in raw.split(","):
        cleaned = value.strip().lower()
        if not cleaned:
            continue
        values.append(None if cleaned in {"none", "any"} else float(cleaned))
    return values


def int_sets(raw: str, base_value: int) -> list[int]:
    if not raw:
        return [base_value]
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def variant_name(settings: ReplaySettings) -> str:
    direction_label = ",".join(sorted(settings.directions)) if settings.directions else "both"
    btc_label = "btcany" if settings.min_btc_24h_return_pct is None else f"btc{settings.min_btc_24h_return_pct:g}"
    btc_72h_label = "btc72any" if settings.max_btc_72h_return_pct is None else f"btc72max{settings.max_btc_72h_return_pct:g}"
    btc_gap_label = "btcgapany" if settings.max_btc_4h_ema_gap_pct is None else f"btcgap{settings.max_btc_4h_ema_gap_pct:g}"
    symbol_return_label = "sym24any" if settings.max_symbol_24h_return_pct is None else f"sym24max{settings.max_symbol_24h_return_pct:g}"
    symbol_atr_label = "atrany" if settings.min_symbol_atr_pct_15m is None else f"atrmin{settings.min_symbol_atr_pct_15m:g}"
    return (
        f"score{settings.min_score:.0f}_{strategy_label(settings.strategies)}_{direction_label}"
        f"_cb{settings.min_minutes_before_close}_ao{settings.min_minutes_after_open}_{btc_label}"
        f"_{btc_72h_label}_{btc_gap_label}_{symbol_return_label}_{symbol_atr_label}"
    )


def weekly_rollup(rows: list[dict[str, object]], weeks: int) -> list[dict[str, object]]:
    weekly_rows = []
    for idx in range(weeks):
        chunk = rows[idx * 7 : (idx + 1) * 7]
        if not chunk:
            continue
        start = float(chunk[0]["starting_equity_usdt"])
        end = float(chunk[-1]["ending_equity_usdt"])
        trades = sum(int(row["closed_trades"]) for row in chunk)
        wins = sum(float(row["win_rate"]) * int(row["closed_trades"]) / 100 for row in chunk)
        weekly_rows.append(
            {
                "week": idx + 1,
                "start_date": chunk[0]["date"],
                "end_date": chunk[-1]["date"],
                "starting_equity_usdt": start,
                "ending_equity_usdt": end,
                "trades": trades,
                "win_rate": wins / trades * 100 if trades else 0.0,
                "pnl_usdt": end - start,
            }
        )
    return weekly_rows


def variant_rank_key(result: VariantResult) -> tuple[float, int, float, float]:
    return (
        profitable_week_count(result),
        aggregate_pnl(result.rows),
        aggregate_win_rate(result.rows),
        trade_count(result.rows),
    )


def aggregate_pnl(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return float(rows[-1]["ending_equity_usdt"]) - float(rows[0]["starting_equity_usdt"])


def aggregate_win_rate(rows: list[dict[str, object]]) -> float:
    trades = trade_count(rows)
    if trades == 0:
        return 0.0
    wins = sum(float(row["win_rate"]) * int(row["closed_trades"]) / 100 for row in rows)
    return wins / trades * 100


def trade_count(rows: list[dict[str, object]]) -> int:
    return sum(int(row["closed_trades"]) for row in rows)


def profitable_week_count(result: VariantResult) -> int:
    return sum(1 for row in result.weekly_rows if float(row["pnl_usdt"]) > 0)


def write_variant_summary(path: Path, results: list[VariantResult], inr_per_usdt: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "profitable_weeks",
                "weeks",
                "trades",
                "win_rate",
                "pnl_usdt",
                "pnl_inr",
                "ending_equity_usdt",
                "settings",
            ]
        )
        for result in results:
            pnl = aggregate_pnl(result.rows)
            writer.writerow(
                [
                    result.name,
                    profitable_week_count(result),
                    len(result.weekly_rows),
                    trade_count(result.rows),
                    f"{aggregate_win_rate(result.rows):.2f}",
                    f"{pnl:.6f}",
                    f"{pnl * inr_per_usdt:.2f}",
                    f"{float(result.rows[-1]['ending_equity_usdt']):.6f}",
                    f"score>={result.settings.min_score}, strategies={strategy_label(result.settings.strategies)}, directions={','.join(sorted(result.settings.directions)) if result.settings.directions else 'both'}, close_buffer={result.settings.min_minutes_before_close}, after_open={result.settings.min_minutes_after_open}, min_btc_24h={result.settings.min_btc_24h_return_pct}, min_btc_21d={result.settings.min_btc_21d_return_pct}, max_short_btc_24h={result.settings.max_btc_24h_return_pct_for_shorts}, max_short_btc_21d={result.settings.max_btc_21d_return_pct_for_shorts}, max_btc_72h={result.settings.max_btc_72h_return_pct}, max_short_btc_72h={result.settings.max_btc_72h_return_pct_for_shorts}, max_btc_4h_gap={result.settings.max_btc_4h_ema_gap_pct}, max_symbol_24h={result.settings.max_symbol_24h_return_pct}, min_symbol_atr={result.settings.min_symbol_atr_pct_15m}",
                ]
            )


def write_weekly_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "week",
                "start_date",
                "end_date",
                "starting_equity_usdt",
                "ending_equity_usdt",
                "trades",
                "win_rate",
                "pnl_usdt",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_report(results: list[VariantResult], output_root: Path, inr_per_usdt: float) -> None:
    print()
    print("Weekly validation results")
    print("variant                                      weeks+  trades  win%    pnl_usdt   pnl_inr")
    for result in results[:10]:
        pnl = aggregate_pnl(result.rows)
        print(
            f"{result.name[:42]:<42} "
            f"{profitable_week_count(result):>2}/{len(result.weekly_rows):<2} "
            f"{trade_count(result.rows):>7} "
            f"{aggregate_win_rate(result.rows):>5.1f} "
            f"{pnl:>10.2f} "
            f"{pnl * inr_per_usdt:>9.2f}"
        )
    best = results[0]
    print()
    print(f"Best by profitable weeks then PnL: {best.name}")
    print(f"Summary CSV: {output_root / 'weekly_variant_summary.csv'}")
    print(f"Best daily CSV: {output_root / 'best_daily_summary.csv'}")
    print(f"Best weekly CSV: {output_root / 'best_weekly_summary.csv'}")
    print(f"Best trades CSV: {output_root / 'best_trades.csv'}")


if __name__ == "__main__":
    main()
