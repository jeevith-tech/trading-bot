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

from institutional_bot.binance_data import BinanceFuturesClient, Candle, frame_from_candles
from institutional_bot.config import ScannerConfig
from institutional_bot.models import MarketSnapshot, RiskState, SetupCandidate
from institutional_bot.paper import PaperConfig, PaperTradingEngine
from institutional_bot.risk import build_position_plan
from institutional_bot.scanner import MarketScanner
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
    strategy_label,
    symbol_atr_pct,
)
from scripts.paper_weekly_binance import float_sets, int_sets, rank_by_historical_volume, selected_days


@dataclass(frozen=True)
class CandidateEvent:
    day: date
    day_start: datetime
    day_end: datetime
    timestamp: datetime
    candidate: SetupCandidate
    btc_24h_return_pct: float
    btc_72h_return_pct: float
    btc_4h_ema_gap_pct: float
    symbol_24h_return_pct: float
    btc_21d_return_pct: float
    symbol_atr_pct_15m: float


@dataclass(frozen=True)
class Variant:
    name: str
    settings: ReplaySettings


def main() -> None:
    args = parse_args()
    tz = load_timezone(args.timezone)
    days = selected_days(args, tz)
    first_start = datetime.combine(days[0], datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)
    final_end = datetime.combine(days[-1] + timedelta(days=1), datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)
    fetch_start = first_start - timedelta(days=args.lookback_days)

    client = BinanceFuturesClient(timeout=args.timeout)
    print(f"Fetching Binance data for {len(days)} days ({days[0]} to {days[-1]}, {args.timezone})...")
    markets = select_markets(client, args)
    base_data: dict[str, SymbolData] = {}
    for offset, market in enumerate(markets, start=1):
        try:
            candles_15m = completed(client.klines(market.symbol, "15m", fetch_start, final_end), "15m", final_end)
            candles_1h = completed(client.klines(market.symbol, "1h", fetch_start, final_end), "1h", final_end)
            candles_4h = completed(client.klines(market.symbol, "4h", fetch_start, final_end), "4h", final_end)
        except Exception as exc:
            print(f"Skipping {market.symbol}: {exc}")
            continue
        if len(candles_15m) < 80 or len(candles_1h) < 40 or len(candles_4h) < 20:
            continue
        replay_market = market
        if args.historical_market_metadata:
            historical_quote_volume = sum(candle.quote_volume for candle in candles_15m if first_start <= candle.open_time < final_end)
            average_daily_quote_volume = historical_quote_volume / max(len(days), 1)
            if average_daily_quote_volume < args.min_volume:
                print(f"Skipping {market.symbol}: historical average daily quote volume below threshold")
                continue
            replay_market = replace(
                market,
                quote_volume=average_daily_quote_volume,
                spread_bps=1.0,
                funding_rate=0.0,
            )
        base_data[market.symbol] = SymbolData(replay_market, candles_15m, candles_1h, candles_4h, 0.0)
        if offset % 10 == 0:
            print(f"Fetched {offset}/{len(markets)} markets...")
    if "BTCUSDT" not in base_data:
        raise RuntimeError("BTCUSDT data is required")
    if args.historical_volume_ranking:
        base_data = rank_by_historical_volume(base_data, first_start, final_end, args.max_symbols)
        print(f"Historical-volume universe: {', '.join(base_data.keys())}")

    min_threshold = min(float_sets(args.thresholds, args.min_score))
    base_settings = ReplaySettings(
        equity_usdt=starting_equity_usdt(args),
        risk_pct=args.risk_pct,
        max_risk_pct=args.max_risk_pct,
        volatility_adjust_risk=not args.fixed_risk,
        min_score=min_threshold,
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
        skip_long_if_btc21_min=args.skip_long_if_btc21_min,
        skip_long_if_btc72_max=args.skip_long_if_btc72_max,
        skip_long_if_symbol24_min=args.skip_long_if_symbol24_min,
        breakeven_trigger_r=args.breakeven_trigger_r,
        soft_stop_r=args.soft_stop_r,
        loss_cooldown_days=args.loss_cooldown_days,
        loss_cooldown_btc_72h_max=args.loss_cooldown_btc_72h_max,
        loss_week_lock_btc_72h_max=args.loss_week_lock_btc_72h_max,
        loss_week_lock_direction=args.loss_week_lock_direction,
        max_trades=args.max_trades_per_day,
    )
    print("Precomputing candidate stream...")
    events_by_day, bars_by_day = precompute_events(base_data, days, tz, base_settings)
    variants = build_variants(args, base_settings)
    print(f"Testing {len(variants)} cached variants...")
    results = [replay_variant(variant, days, events_by_day, bars_by_day, args.inr_per_usdt) for variant in variants]
    results.sort(key=rank_key, reverse=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_variant_summary(output_dir / "variant_summary.csv", results)
    best = results[0]
    write_daily_summary(output_dir / "best_daily_summary.csv", best["daily_rows"])
    write_trades(output_dir / "best_trades.csv", best["trades"])
    print_report(results, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast cached optimizer for one-trade/day Binance replay filters.")
    parser.add_argument("--timezone", default="Asia/Calcutta")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--candidate-symbols", type=int, default=20)
    parser.add_argument("--symbols", default="", help="Comma-separated fixed Binance symbols for fair repeatable replays.")
    parser.add_argument("--historical-volume-ranking", action="store_true")
    parser.add_argument("--historical-market-metadata", action="store_true", help="Use replay-window candle volume and fixed spread/funding assumptions for deterministic historical tests.")
    parser.add_argument("--min-volume", type=float, default=50_000_000)
    parser.add_argument("--equity", type=float, default=100_000)
    parser.add_argument("--capital-inr", type=float, default=None)
    parser.add_argument("--inr-per-usdt", type=float, default=95.0)
    parser.add_argument("--risk-pct", type=float, default=10.0)
    parser.add_argument("--max-risk-pct", type=float, default=10.0)
    parser.add_argument("--fixed-risk", action="store_true")
    parser.add_argument("--min-score", type=float, default=85.0)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--max-trades-per-day", type=int, default=1)
    parser.add_argument("--strategies", default="trend_continuation")
    parser.add_argument("--directions", default="long")
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
    parser.add_argument("--skip-long-if-btc21-min", type=float, default=None)
    parser.add_argument("--skip-long-if-btc72-max", type=float, default=None)
    parser.add_argument("--skip-long-if-symbol24-min", type=float, default=None)
    parser.add_argument("--breakeven-trigger-r", type=float, default=None)
    parser.add_argument("--soft-stop-r", type=float, default=None)
    parser.add_argument("--loss-cooldown-days", type=int, default=0)
    parser.add_argument("--loss-cooldown-btc-72h-max", type=float, default=None)
    parser.add_argument("--loss-week-lock-btc-72h-max", type=float, default=None)
    parser.add_argument("--loss-week-lock-direction", default="")
    parser.add_argument("--thresholds", default="85")
    parser.add_argument("--btc-return-sets", default="none")
    parser.add_argument("--btc-21d-min-sets", default="")
    parser.add_argument("--short-btc-24h-max-sets", default="")
    parser.add_argument("--short-btc-21d-max-sets", default="")
    parser.add_argument("--btc-72h-max-sets", default="none")
    parser.add_argument("--short-btc-72h-max-sets", default="")
    parser.add_argument("--btc-4h-ema-gap-max-sets", default="none")
    parser.add_argument("--symbol-24h-max-sets", default="none")
    parser.add_argument("--symbol-atr-min-sets", default="none")
    parser.add_argument("--skip-long-btc21-min-sets", default="none")
    parser.add_argument("--skip-long-btc72-max-sets", default="none")
    parser.add_argument("--skip-long-symbol24-min-sets", default="none")
    parser.add_argument("--breakeven-trigger-r-sets", default="none")
    parser.add_argument("--soft-stop-r-sets", default="none")
    parser.add_argument("--after-open-sets", default="0")
    parser.add_argument("--loss-cooldown-day-sets", default="0")
    parser.add_argument("--loss-cooldown-btc-72h-max-sets", default="none")
    parser.add_argument("--loss-week-lock-btc-72h-max-sets", default="none")
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--output-dir", default="reports/fast_filter_optimizer")
    return parser.parse_args()


def select_markets(client: BinanceFuturesClient, args: argparse.Namespace):
    if not args.symbols:
        return client.top_usdt_perp_markets(
            limit=max(args.max_symbols, args.candidate_symbols),
            min_quote_volume=args.min_volume,
        )

    requested = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    all_markets = client.top_usdt_perp_markets(limit=max(500, len(requested)), min_quote_volume=0)
    by_symbol = {market.symbol: market for market in all_markets}
    selected = []
    missing = []
    too_small = []
    for symbol in requested:
        market = by_symbol.get(symbol)
        if market is None:
            missing.append(symbol)
            continue
        if market.quote_volume < args.min_volume:
            too_small.append(symbol)
            continue
        selected.append(market)
    if missing:
        print(f"Fixed-symbol universe missing from Binance eligibility: {', '.join(missing)}")
    if too_small:
        print(f"Fixed-symbol universe below min volume: {', '.join(too_small)}")
    if not selected:
        raise RuntimeError("no requested fixed symbols passed Binance liquidity filters")
    print(f"Using fixed-symbol universe: {', '.join(market.symbol for market in selected)}")
    return selected


def precompute_events(
    data: dict[str, SymbolData],
    days: list[date],
    tz,
    base_settings: ReplaySettings,
) -> tuple[dict[date, dict[datetime, list[CandidateEvent]]], dict[date, dict[tuple[str, datetime], Candle]]]:
    scanner = MarketScanner(
        ScannerConfig(
            min_daily_volume_usd=base_settings.min_volume,
            min_score_to_trade=base_settings.min_score,
            risk_per_trade_pct=base_settings.risk_pct,
            max_risk_per_trade_pct=base_settings.max_risk_pct,
            volatility_adjust_risk=False,
        )
    )
    neutral_risk = RiskState(equity=base_settings.equity_usdt)
    events_by_day: dict[date, dict[datetime, list[CandidateEvent]]] = {}
    bars_by_day: dict[date, dict[tuple[str, datetime], Candle]] = {}
    for day in days:
        day_start = datetime.combine(day, datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)
        day_end = day_start + timedelta(days=1)
        btc_times = [candle.open_time for candle in data["BTCUSDT"].candles_15m if day_start <= candle.open_time < day_end]
        day_events: dict[datetime, list[CandidateEvent]] = {}
        day_bars: dict[tuple[str, datetime], Candle] = {}
        for timestamp in btc_times:
            snapshots = []
            for symbol, symbol_data in data.items():
                index_15m = index_at_or_before(symbol_data.candles_15m, timestamp)
                if index_15m is None:
                    continue
                current = symbol_data.candles_15m[index_15m]
                day_bars[(symbol, timestamp)] = current
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
                        open_interest_change_pct=symbol_data.oi_change_pct,
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
            decision = scanner.scan(snapshots, neutral_risk)
            event_list = []
            for candidate, _ in decision.tradable:
                symbol_data = data[candidate.symbol]
                event_list.append(
                    CandidateEvent(
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
                )
            if event_list:
                day_events[timestamp] = sorted(event_list, key=lambda event: event.candidate.score, reverse=True)
        events_by_day[day] = day_events
        bars_by_day[day] = day_bars
        print(f"  {day}: cached {sum(len(events) for events in day_events.values())} tradable candidate events")
    return events_by_day, bars_by_day


def build_variants(args: argparse.Namespace, base: ReplaySettings) -> list[Variant]:
    variants = []
    for soft_stop_r in float_sets(args.soft_stop_r_sets, base.soft_stop_r):
        for breakeven_trigger_r in float_sets(args.breakeven_trigger_r_sets, base.breakeven_trigger_r):
            for week_lock_btc_72h_max in float_sets(args.loss_week_lock_btc_72h_max_sets, base.loss_week_lock_btc_72h_max):
                for cooldown_btc_72h_max in float_sets(args.loss_cooldown_btc_72h_max_sets, base.loss_cooldown_btc_72h_max):
                    for skip_symbol24_min in float_sets(args.skip_long_symbol24_min_sets, base.skip_long_if_symbol24_min):
                        for skip_btc72_max in float_sets(args.skip_long_btc72_max_sets, base.skip_long_if_btc72_max):
                            for skip_btc21_min in float_sets(args.skip_long_btc21_min_sets, base.skip_long_if_btc21_min):
                                for atr_min in float_sets(args.symbol_atr_min_sets, base.min_symbol_atr_pct_15m):
                                    for symbol_24h_max in float_sets(args.symbol_24h_max_sets, base.max_symbol_24h_return_pct):
                                        for btc_gap_max in float_sets(args.btc_4h_ema_gap_max_sets, base.max_btc_4h_ema_gap_pct):
                                            for btc_72h_max in float_sets(args.btc_72h_max_sets, base.max_btc_72h_return_pct):
                                                for short_btc_72h_max in float_sets(args.short_btc_72h_max_sets, base.max_btc_72h_return_pct_for_shorts):
                                                    for short_btc_21d_max in float_sets(args.short_btc_21d_max_sets, base.max_btc_21d_return_pct_for_shorts):
                                                        for short_btc_24h_max in float_sets(args.short_btc_24h_max_sets, base.max_btc_24h_return_pct_for_shorts):
                                                            for btc_21d_min in float_sets(args.btc_21d_min_sets, base.min_btc_21d_return_pct):
                                                                for btc_24h_min in float_sets(args.btc_return_sets, base.min_btc_24h_return_pct):
                                                                    for cooldown_days in int_sets(args.loss_cooldown_day_sets, base.loss_cooldown_days):
                                                                        for after_open in int_sets(args.after_open_sets, base.min_minutes_after_open):
                                                                            for threshold in float_sets(args.thresholds, base.min_score):
                                                                                settings = replace(
                                                                                    base,
                                                                                    min_score=threshold,
                                                                                    min_minutes_after_open=after_open,
                                                                                    min_btc_24h_return_pct=btc_24h_min,
                                                                                    min_btc_21d_return_pct=btc_21d_min,
                                                                                    max_btc_24h_return_pct_for_shorts=short_btc_24h_max,
                                                                                    max_btc_21d_return_pct_for_shorts=short_btc_21d_max,
                                                                                    max_btc_72h_return_pct_for_shorts=short_btc_72h_max,
                                                                                    max_btc_72h_return_pct=btc_72h_max,
                                                                                    max_btc_4h_ema_gap_pct=btc_gap_max,
                                                                                    max_symbol_24h_return_pct=symbol_24h_max,
                                                                                    min_symbol_atr_pct_15m=atr_min,
                                                                                    skip_long_if_btc21_min=skip_btc21_min,
                                                                                    skip_long_if_btc72_max=skip_btc72_max,
                                                                                    skip_long_if_symbol24_min=skip_symbol24_min,
                                                                                    breakeven_trigger_r=breakeven_trigger_r,
                                                                                    soft_stop_r=soft_stop_r,
                                                                                    loss_cooldown_days=cooldown_days,
                                                                                    loss_cooldown_btc_72h_max=cooldown_btc_72h_max,
                                                                                    loss_week_lock_btc_72h_max=week_lock_btc_72h_max,
                                                                                )
                                                                                variants.append(Variant(name=variant_name(settings), settings=settings))
    return variants


def replay_variant(
    variant: Variant,
    days: list[date],
    events_by_day: dict[date, dict[datetime, list[CandidateEvent]]],
    bars_by_day: dict[date, dict[tuple[str, datetime], Candle]],
    inr_per_usdt: float,
) -> dict[str, object]:
    current_equity = variant.settings.equity_usdt
    daily_rows: list[dict[str, object]] = []
    trade_rows: list[list[str]] = []
    cooldown_remaining = 0
    week_lock_until_idx = -1
    for day_index, day in enumerate(days):
        timestamps = sorted({timestamp for _, timestamp in bars_by_day[day]})
        if not timestamps:
            continue
        if cooldown_remaining > 0 or day_index <= week_lock_until_idx:
            daily_rows.append(
                {
                    "date": str(day),
                    "starting_equity_usdt": current_equity,
                    "ending_equity_usdt": current_equity,
                    "signals": 0,
                    "closed_trades": 0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "pnl_usdt": 0.0,
                    "pnl_inr": 0.0,
                    "pnl_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                }
            )
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            continue
        engine = PaperTradingEngine(
            PaperConfig(
                starting_equity=current_equity,
                fee_bps=variant.settings.fee_bps,
                slippage_bps=variant.settings.slippage_bps,
                max_concurrent_positions=variant.settings.max_positions,
                breakeven_trigger_r=variant.settings.breakeven_trigger_r,
                soft_stop_r=variant.settings.soft_stop_r,
            )
        )
        signal_count = 0
        opened_symbols: set[str] = set()
        opened_events: dict[tuple[str, datetime], CandidateEvent] = {}
        last_marks: dict[str, float] = {}
        for timestamp in timestamps:
            for position in [position for position in engine.positions if position.is_open]:
                candle = bars_by_day[day].get((position.candidate.symbol, timestamp))
                if candle is not None:
                    last_marks[position.candidate.symbol] = candle.close
                    engine.update_bar(position.candidate.symbol, timestamp, candle.high, candle.low, candle.close)
            if variant.settings.max_trades is not None and signal_count >= variant.settings.max_trades:
                continue
            if any(position.is_open for position in engine.positions):
                continue
            for event in events_by_day[day].get(timestamp, []):
                candidate = event.candidate
                if candidate.symbol in opened_symbols:
                    continue
                if not event_allowed(event, variant.settings):
                    continue
                plan = build_position_plan(candidate, engine.risk_state(), variant.settings.risk_pct)
                engine.open_position(candidate, plan, timestamp)
                opened_events[(candidate.symbol, timestamp)] = event
                opened_symbols.add(candidate.symbol)
                signal_count += 1
                break
        for (symbol, timestamp), candle in bars_by_day[day].items():
            if timestamp == timestamps[-1]:
                last_marks[symbol] = candle.close
        engine.close_all(timestamps[-1], last_marks, reason="session_end")
        summary = engine.summary(last_marks)
        current_equity = summary.equity
        if loss_cooldown_triggered(summary.total_pnl, engine.trades, opened_events, variant.settings):
            cooldown_remaining = variant.settings.loss_cooldown_days
        if loss_week_lock_triggered(summary.total_pnl, engine.trades, opened_events, variant.settings):
            week_lock_until_idx = max(week_lock_until_idx, ((day_index // 7) + 1) * 7 - 1)
        daily_rows.append(
            {
                "date": str(day),
                "starting_equity_usdt": summary.starting_equity,
                "ending_equity_usdt": summary.equity,
                "signals": signal_count,
                "closed_trades": summary.closed_trades,
                "win_rate": summary.win_rate,
                "profit_factor": summary.profit_factor,
                "pnl_usdt": summary.total_pnl,
                "pnl_inr": summary.total_pnl * inr_per_usdt,
                "pnl_pct": summary.total_pnl_pct,
                "max_drawdown_pct": summary.max_drawdown_pct,
            }
        )
        for trade in engine.trades:
            event = opened_events.get((trade.symbol, trade.opened_at))
            trade_rows.append(
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
                    "" if event is None else f"{event.btc_24h_return_pct:.4f}",
                    "" if event is None else f"{event.btc_72h_return_pct:.4f}",
                    "" if event is None else f"{event.btc_21d_return_pct:.4f}",
                    "" if event is None else f"{event.btc_4h_ema_gap_pct:.4f}",
                    "" if event is None else f"{event.symbol_24h_return_pct:.4f}",
                    "" if event is None else f"{event.symbol_atr_pct_15m:.4f}",
                ]
            )
    return {
        "variant": variant.name,
        "settings": variant.settings,
        "daily_rows": daily_rows,
        "trades": trade_rows,
        "trade_count": sum(int(row["closed_trades"]) for row in daily_rows),
        "win_rate": aggregate_win_rate(daily_rows),
        "pnl_usdt": current_equity - variant.settings.equity_usdt,
        "pnl_inr": (current_equity - variant.settings.equity_usdt) * inr_per_usdt,
        "ending_equity_usdt": current_equity,
        "profitable_weeks": profitable_week_count(daily_rows),
        "weeks": (len(daily_rows) + 6) // 7,
    }


def loss_cooldown_triggered(
    total_pnl: float,
    trades,
    opened_events: dict[tuple[str, datetime], CandidateEvent],
    settings: ReplaySettings,
) -> bool:
    if total_pnl >= 0 or settings.loss_cooldown_days <= 0:
        return False
    if settings.loss_cooldown_btc_72h_max is None:
        return True
    for trade in trades:
        if trade.realized_pnl >= 0:
            continue
        event = opened_events.get((trade.symbol, trade.opened_at))
        if event is not None and event.btc_72h_return_pct <= settings.loss_cooldown_btc_72h_max:
            return True
    return False


def loss_week_lock_triggered(
    total_pnl: float,
    trades,
    opened_events: dict[tuple[str, datetime], CandidateEvent],
    settings: ReplaySettings,
) -> bool:
    if total_pnl >= 0 or settings.loss_week_lock_btc_72h_max is None:
        return False
    lock_direction = settings.loss_week_lock_direction.strip().lower()
    for trade in trades:
        if trade.realized_pnl >= 0:
            continue
        event = opened_events.get((trade.symbol, trade.opened_at))
        if event is None:
            continue
        if lock_direction and event.candidate.direction.value != lock_direction:
            continue
        if event.btc_72h_return_pct <= settings.loss_week_lock_btc_72h_max:
            return True
    return False


def event_allowed(event: CandidateEvent, settings: ReplaySettings) -> bool:
    candidate = event.candidate
    if candidate.score < settings.min_score:
        return False
    if settings.strategies and candidate.strategy.value not in settings.strategies:
        return False
    if settings.directions and candidate.direction.value not in settings.directions:
        return False
    if settings.min_minutes_after_open and event.timestamp < event.day_start + timedelta(minutes=settings.min_minutes_after_open):
        return False
    if settings.min_minutes_before_close and event.timestamp > event.day_end - timedelta(minutes=settings.min_minutes_before_close):
        return False
    if candidate.direction.value == "long":
        if (
            settings.skip_long_if_btc21_min is not None
            and settings.skip_long_if_btc72_max is not None
            and settings.skip_long_if_symbol24_min is not None
            and event.btc_21d_return_pct >= settings.skip_long_if_btc21_min
            and event.btc_72h_return_pct <= settings.skip_long_if_btc72_max
            and event.symbol_24h_return_pct >= settings.skip_long_if_symbol24_min
        ):
            return False
        if settings.min_btc_24h_return_pct is not None and event.btc_24h_return_pct < settings.min_btc_24h_return_pct:
            return False
        if settings.min_btc_21d_return_pct is not None and event.btc_21d_return_pct < settings.min_btc_21d_return_pct:
            return False
        if settings.max_btc_72h_return_pct is not None and event.btc_72h_return_pct > settings.max_btc_72h_return_pct:
            return False
        if settings.max_btc_4h_ema_gap_pct is not None and event.btc_4h_ema_gap_pct > settings.max_btc_4h_ema_gap_pct:
            return False
        if settings.max_symbol_24h_return_pct is not None and event.symbol_24h_return_pct > settings.max_symbol_24h_return_pct:
            return False
        if settings.min_symbol_atr_pct_15m is not None and event.symbol_atr_pct_15m < settings.min_symbol_atr_pct_15m:
            return False
    if candidate.direction.value == "short":
        if settings.max_btc_24h_return_pct_for_shorts is not None and event.btc_24h_return_pct > settings.max_btc_24h_return_pct_for_shorts:
            return False
        if settings.max_btc_21d_return_pct_for_shorts is not None and event.btc_21d_return_pct > settings.max_btc_21d_return_pct_for_shorts:
            return False
        if settings.max_btc_72h_return_pct_for_shorts is not None and event.btc_72h_return_pct > settings.max_btc_72h_return_pct_for_shorts:
            return False
    return True


def rank_key(result: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        float(result["profitable_weeks"]),
        float(result["pnl_usdt"]),
        float(result["win_rate"]),
        int(result["trade_count"]),
    )


def aggregate_win_rate(rows: list[dict[str, object]]) -> float:
    trades = sum(int(row["closed_trades"]) for row in rows)
    if not trades:
        return 0.0
    wins = sum(float(row["win_rate"]) * int(row["closed_trades"]) / 100 for row in rows)
    return wins / trades * 100


def profitable_week_count(rows: list[dict[str, object]]) -> int:
    count = 0
    for idx in range((len(rows) + 6) // 7):
        chunk = rows[idx * 7 : (idx + 1) * 7]
        if sum(float(row["pnl_usdt"]) for row in chunk) > 0:
            count += 1
    return count


def variant_name(settings: ReplaySettings) -> str:
    btc24 = "btc24any" if settings.min_btc_24h_return_pct is None else f"btc24min{settings.min_btc_24h_return_pct:g}"
    btc21 = "btc21any" if settings.min_btc_21d_return_pct is None else f"btc21min{settings.min_btc_21d_return_pct:g}"
    short_btc24 = "sbtc24any" if settings.max_btc_24h_return_pct_for_shorts is None else f"sbtc24max{settings.max_btc_24h_return_pct_for_shorts:g}"
    short_btc21 = "sbtc21any" if settings.max_btc_21d_return_pct_for_shorts is None else f"sbtc21max{settings.max_btc_21d_return_pct_for_shorts:g}"
    btc72 = "btc72any" if settings.max_btc_72h_return_pct is None else f"btc72max{settings.max_btc_72h_return_pct:g}"
    short_btc72 = "sbtc72any" if settings.max_btc_72h_return_pct_for_shorts is None else f"sbtc72max{settings.max_btc_72h_return_pct_for_shorts:g}"
    btcgap = "btcgapany" if settings.max_btc_4h_ema_gap_pct is None else f"btcgapmax{settings.max_btc_4h_ema_gap_pct:g}"
    sym24 = "sym24any" if settings.max_symbol_24h_return_pct is None else f"sym24max{settings.max_symbol_24h_return_pct:g}"
    atr = "atrany" if settings.min_symbol_atr_pct_15m is None else f"atrmin{settings.min_symbol_atr_pct_15m:g}"
    skip_long = (
        "slany"
        if settings.skip_long_if_btc21_min is None or settings.skip_long_if_btc72_max is None or settings.skip_long_if_symbol24_min is None
        else f"slb21min{settings.skip_long_if_btc21_min:g}b72max{settings.skip_long_if_btc72_max:g}s24min{settings.skip_long_if_symbol24_min:g}"
    )
    cooldown = f"cd{settings.loss_cooldown_days}"
    cooldown_filter = "cdbtc72any" if settings.loss_cooldown_btc_72h_max is None else f"cdbtc72max{settings.loss_cooldown_btc_72h_max:g}"
    week_lock = "wlany" if settings.loss_week_lock_btc_72h_max is None else f"wl{settings.loss_week_lock_direction or 'any'}btc72max{settings.loss_week_lock_btc_72h_max:g}"
    breakeven = "beany" if settings.breakeven_trigger_r is None else f"be{settings.breakeven_trigger_r:g}r"
    soft_stop = "ssany" if settings.soft_stop_r is None else f"ss{settings.soft_stop_r:g}r"
    directions = ",".join(sorted(settings.directions)) if settings.directions else "both"
    return f"score{settings.min_score:g}_{strategy_label(settings.strategies)}_{directions}_ao{settings.min_minutes_after_open}_{cooldown}_{cooldown_filter}_{week_lock}_{breakeven}_{soft_stop}_{btc24}_{btc21}_{short_btc24}_{short_btc21}_{btc72}_{short_btc72}_{btcgap}_{sym24}_{atr}_{skip_long}"


def write_variant_summary(path: Path, results: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "profitable_weeks", "weeks", "trades", "win_rate", "pnl_usdt", "pnl_inr", "ending_equity_usdt", "settings"])
        for result in results:
            settings: ReplaySettings = result["settings"]
            writer.writerow(
                [
                    result["variant"],
                    result["profitable_weeks"],
                    result["weeks"],
                    result["trade_count"],
                    f"{float(result['win_rate']):.2f}",
                    f"{float(result['pnl_usdt']):.6f}",
                    f"{float(result['pnl_inr']):.2f}",
                    f"{float(result['ending_equity_usdt']):.6f}",
                    f"score>={settings.min_score}, strategies={strategy_label(settings.strategies)}, directions={','.join(sorted(settings.directions)) if settings.directions else 'both'}, after_open={settings.min_minutes_after_open}, loss_cooldown_days={settings.loss_cooldown_days}, loss_cooldown_btc_72h_max={settings.loss_cooldown_btc_72h_max}, loss_week_lock_btc_72h_max={settings.loss_week_lock_btc_72h_max}, loss_week_lock_direction={settings.loss_week_lock_direction}, breakeven_trigger_r={settings.breakeven_trigger_r}, soft_stop_r={settings.soft_stop_r}, min_btc_24h={settings.min_btc_24h_return_pct}, min_btc_21d={settings.min_btc_21d_return_pct}, max_short_btc_24h={settings.max_btc_24h_return_pct_for_shorts}, max_short_btc_21d={settings.max_btc_21d_return_pct_for_shorts}, max_btc_72h={settings.max_btc_72h_return_pct}, max_short_btc_72h={settings.max_btc_72h_return_pct_for_shorts}, max_btc_4h_gap={settings.max_btc_4h_ema_gap_pct}, max_symbol_24h={settings.max_symbol_24h_return_pct}, min_symbol_atr={settings.min_symbol_atr_pct_15m}, skip_long_if_btc21_min={settings.skip_long_if_btc21_min}, skip_long_if_btc72_max={settings.skip_long_if_btc72_max}, skip_long_if_symbol24_min={settings.skip_long_if_symbol24_min}",
                ]
            )


def write_daily_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_trades(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "date",
                "symbol",
                "strategy",
                "direction",
                "score",
                "opened_at",
                "closed_at",
                "realized_pnl",
                "exit_reason",
                "btc_24h_return_pct",
                "btc_72h_return_pct",
                "btc_21d_return_pct",
                "btc_4h_ema_gap_pct",
                "symbol_24h_return_pct",
                "symbol_atr_pct_15m",
            ]
        )
        writer.writerows(rows)


def print_report(results: list[dict[str, object]], output_dir: Path) -> None:
    print()
    print("Fast filter optimizer results")
    print("variant                                      weeks+  trades  win%    pnl_usdt   pnl_inr")
    for result in results[:12]:
        print(
            f"{str(result['variant'])[:42]:<42} "
            f"{int(result['profitable_weeks']):>2}/{int(result['weeks']):<2} "
            f"{int(result['trade_count']):>7} "
            f"{float(result['win_rate']):>5.1f} "
            f"{float(result['pnl_usdt']):>10.2f} "
            f"{float(result['pnl_inr']):>9.2f}"
        )
    print()
    print(f"Best: {results[0]['variant']}")
    print(f"Summary CSV: {output_dir / 'variant_summary.csv'}")
    print(f"Best daily CSV: {output_dir / 'best_daily_summary.csv'}")
    print(f"Best trades CSV: {output_dir / 'best_trades.csv'}")


if __name__ == "__main__":
    main()
