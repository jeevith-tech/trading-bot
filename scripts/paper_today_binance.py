from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from math import sqrt
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from institutional_bot.binance_data import BinanceFuturesClient, BinanceMarket, Candle, frame_from_candles
from institutional_bot.config import ScannerConfig
from institutional_bot.indicators import atr, ema
from institutional_bot.models import MarketSnapshot
from institutional_bot.paper import PaperConfig, PaperTradingEngine
from institutional_bot.scanner import MarketScanner


@dataclass(frozen=True)
class SymbolData:
    market: BinanceMarket
    candles_15m: list[Candle]
    candles_1h: list[Candle]
    candles_4h: list[Candle]
    oi_change_pct: float


@dataclass(frozen=True)
class ReplaySettings:
    equity_usdt: float
    risk_pct: float
    max_risk_pct: float
    volatility_adjust_risk: bool
    min_score: float
    min_volume: float
    fee_bps: float
    slippage_bps: float
    max_positions: int
    flat_at_end: bool
    strategies: frozenset[str]
    directions: frozenset[str]
    min_minutes_before_close: int = 0
    min_minutes_after_open: int = 0
    min_btc_24h_return_pct: float | None = None
    min_btc_21d_return_pct: float | None = None
    max_btc_24h_return_pct_for_shorts: float | None = None
    max_btc_21d_return_pct_for_shorts: float | None = None
    max_btc_72h_return_pct: float | None = None
    max_btc_72h_return_pct_for_shorts: float | None = None
    max_btc_4h_ema_gap_pct: float | None = None
    max_symbol_24h_return_pct: float | None = None
    min_symbol_atr_pct_15m: float | None = None
    skip_long_if_btc21_min: float | None = None
    skip_long_if_btc72_max: float | None = None
    skip_long_if_symbol24_min: float | None = None
    breakeven_trigger_r: float | None = None
    soft_stop_r: float | None = None
    loss_cooldown_days: int = 0
    loss_cooldown_btc_72h_max: float | None = None
    loss_week_lock_btc_72h_max: float | None = None
    loss_week_lock_direction: str = ""
    max_trades: int | None = None


@dataclass(frozen=True)
class ReplayResult:
    settings: ReplaySettings
    engine: PaperTradingEngine
    signal_count: int
    marks: dict[str, float]
    summary: object


def main() -> None:
    args = parse_args()
    tz = load_timezone(args.timezone)
    now_utc = datetime.now(timezone.utc)
    today_start_local = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_utc = today_start_local.astimezone(timezone.utc)
    lookback_start = today_start_utc - timedelta(days=args.lookback_days)

    client = BinanceFuturesClient(timeout=args.timeout)
    print(f"Fetching Binance USDT-M futures data for {today_start_local.date()} ({args.timezone})...")
    markets = client.top_usdt_perp_markets(limit=args.max_symbols, min_quote_volume=args.min_volume)
    if not markets:
        raise RuntimeError("no markets passed the Binance liquidity filters")
    print(f"Selected {len(markets)} liquid markets. Top symbols: {', '.join(market.symbol for market in markets[:10])}")

    data: dict[str, SymbolData] = {}
    for offset, market in enumerate(markets, start=1):
        try:
            candles_15m = completed(client.klines(market.symbol, "15m", lookback_start, now_utc), "15m", now_utc)
            candles_1h = completed(client.klines(market.symbol, "1h", lookback_start, now_utc), "1h", now_utc)
            candles_4h = completed(client.klines(market.symbol, "4h", lookback_start, now_utc), "4h", now_utc)
            oi_change = client.open_interest_change_pct(market.symbol, today_start_utc, now_utc)
        except Exception as exc:
            print(f"Skipping {market.symbol}: {exc}")
            continue
        if len(candles_15m) < 80 or len(candles_1h) < 40 or len(candles_4h) < 20:
            continue
        data[market.symbol] = SymbolData(market, candles_15m, candles_1h, candles_4h, oi_change)
        if offset % 20 == 0:
            print(f"Fetched {offset}/{len(markets)} markets...")

    if "BTCUSDT" not in data:
        raise RuntimeError("BTCUSDT data is required for correlation scoring")

    equity_usdt = starting_equity_usdt(args)
    base_settings = ReplaySettings(
        equity_usdt=equity_usdt,
        risk_pct=args.risk_pct,
        max_risk_pct=args.max_risk_pct,
        volatility_adjust_risk=not args.fixed_risk,
        min_score=args.min_score,
        min_volume=args.min_volume,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        max_positions=args.max_positions,
        flat_at_end=args.flat_at_end,
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
        max_trades=args.max_trades,
    )

    if args.optimize_winrate:
        result = optimize_winrate(data, today_start_utc, now_utc, base_settings, args.min_optimizer_trades)
        print_optimizer_table(result["ranked"])
        best = result["best"]
        print()
        print("Best conservative win-rate filter selected.")
        print(
            f"Strategies={strategy_label(best.settings.strategies)} "
            f"min_score={best.settings.min_score:.1f} "
            f"max_positions={best.settings.max_positions}"
        )
        replay = best
    else:
        replay = run_replay(data, today_start_utc, now_utc, base_settings)

    engine = replay.engine
    summary = replay.summary
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trades(output_path, engine)
    print_report(summary, engine, replay.signal_count, today_start_local, output_path, args.inr_per_usdt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay today's Binance futures candles in paper-trading mode.")
    parser.add_argument("--timezone", default="Asia/Calcutta")
    parser.add_argument("--max-symbols", type=int, default=140)
    parser.add_argument("--min-volume", type=float, default=50_000_000)
    parser.add_argument("--equity", type=float, default=100_000)
    parser.add_argument("--capital-inr", type=float, default=None)
    parser.add_argument("--inr-per-usdt", type=float, default=95.0)
    parser.add_argument("--risk-pct", type=float, default=10.0)
    parser.add_argument("--max-risk-pct", type=float, default=10.0)
    parser.add_argument("--fixed-risk", action="store_true", help="Use exactly --risk-pct per trade instead of volatility-reducing risk.")
    parser.add_argument("--min-score", type=float, default=85.0)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--flat-at-end", action="store_true", help="Close open positions at the latest available mark.")
    parser.add_argument("--strategies", default="", help="Comma-separated strategy filter, for example momentum_breakout,liquidity_sweep.")
    parser.add_argument("--directions", default="", help="Comma-separated direction filter: long,short.")
    parser.add_argument("--min-minutes-before-close", type=int, default=0, help="Do not open new trades too close to replay session end.")
    parser.add_argument("--min-minutes-after-open", type=int, default=0, help="Do not open new trades too close to replay session start.")
    parser.add_argument("--min-btc-24h-return-pct", type=float, default=None, help="For longs, require BTC's prior 24h return to be at least this value.")
    parser.add_argument("--min-btc-21d-return-pct", type=float, default=None, help="For longs, require BTC's prior 21d return to be at least this value.")
    parser.add_argument("--max-btc-24h-return-pct-for-shorts", type=float, default=None, help="For shorts, require BTC's prior 24h return to be no more than this value.")
    parser.add_argument("--max-btc-21d-return-pct-for-shorts", type=float, default=None, help="For shorts, require BTC's prior 21d return to be no more than this value.")
    parser.add_argument("--max-btc-72h-return-pct", type=float, default=None, help="For longs, skip when BTC is overextended over the prior 72h.")
    parser.add_argument("--max-btc-72h-return-pct-for-shorts", type=float, default=None, help="For shorts, require BTC's prior 72h return to be no more than this value.")
    parser.add_argument("--max-btc-4h-ema-gap-pct", type=float, default=None, help="For longs, skip when BTC 4h EMA20 is too far above EMA50.")
    parser.add_argument("--max-symbol-24h-return-pct", type=float, default=None, help="For longs, skip symbols already overextended over the prior 24h.")
    parser.add_argument("--min-symbol-atr-pct-15m", type=float, default=None, help="For longs, require enough 15m ATR percent so the move can pay fees and risk.")
    parser.add_argument("--skip-long-if-btc21-min", type=float, default=None, help="Skip longs when BTC 21d return, BTC 72h return, and symbol 24h return match the late-long trap filter.")
    parser.add_argument("--skip-long-if-btc72-max", type=float, default=None, help="BTC 72h return ceiling used with the late-long trap filter.")
    parser.add_argument("--skip-long-if-symbol24-min", type=float, default=None, help="Symbol 24h return floor used with the late-long trap filter.")
    parser.add_argument("--breakeven-trigger-r", type=float, default=None, help="Move stop to breakeven once price reaches this R multiple before TP1.")
    parser.add_argument("--soft-stop-r", type=float, default=None, help="Close at bar close when unrealized loss reaches this R multiple before the hard stop.")
    parser.add_argument("--max-trades", type=int, default=None, help="Maximum trades to open during this replay.")
    parser.add_argument("--one-trade-per-day", action="store_true", help="Alias for --max-trades 1 in a single-day replay.")
    parser.add_argument("--optimize-winrate", action="store_true", help="Grid-search conservative filters on today's replay data.")
    parser.add_argument("--min-optimizer-trades", type=int, default=5)
    parser.add_argument("--output", default="reports/binance_paper_today_trades.csv")
    args = parser.parse_args()
    if args.one_trade_per_day:
        args.max_trades = 1
    return args


def run_replay(
    data: dict[str, SymbolData],
    today_start_utc: datetime,
    now_utc: datetime,
    settings: ReplaySettings,
) -> ReplayResult:
    engine = PaperTradingEngine(
        PaperConfig(
            starting_equity=settings.equity_usdt,
            fee_bps=settings.fee_bps,
            slippage_bps=settings.slippage_bps,
            max_concurrent_positions=settings.max_positions,
            breakeven_trigger_r=settings.breakeven_trigger_r,
            soft_stop_r=settings.soft_stop_r,
        )
    )
    scanner = MarketScanner(
        ScannerConfig(
            min_daily_volume_usd=settings.min_volume,
            min_score_to_trade=settings.min_score,
            risk_per_trade_pct=settings.risk_pct,
            max_risk_per_trade_pct=settings.max_risk_pct,
            volatility_adjust_risk=settings.volatility_adjust_risk,
        )
    )

    btc_today_times = [
        candle.open_time
        for candle in data["BTCUSDT"].candles_15m
        if today_start_utc <= candle.open_time < now_utc
    ]
    if not btc_today_times:
        raise RuntimeError("no completed 15m candles are available for today yet")

    last_marks: dict[str, float] = {}
    signal_count = 0
    for timestamp in btc_today_times:
        snapshots: list[MarketSnapshot] = []
        for symbol, symbol_data in data.items():
            index_15m = index_at_or_before(symbol_data.candles_15m, timestamp)
            if index_15m is None:
                continue
            current = symbol_data.candles_15m[index_15m]
            last_marks[symbol] = current.close
            engine.update_bar(symbol, timestamp, current.high, current.low, current.close)

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

        decision = scanner.scan(snapshots, engine.risk_state())
        for candidate, plan in decision.tradable:
            if settings.max_trades is not None and signal_count >= settings.max_trades:
                continue
            if settings.strategies and candidate.strategy.value not in settings.strategies:
                continue
            if settings.directions and candidate.direction.value not in settings.directions:
                continue
            if settings.min_minutes_after_open and timestamp < today_start_utc + timedelta(minutes=settings.min_minutes_after_open):
                continue
            if settings.min_minutes_before_close and timestamp > now_utc - timedelta(minutes=settings.min_minutes_before_close):
                continue
            if candidate.direction.value == "long" and not long_context_allowed(data, data[candidate.symbol], timestamp, settings):
                continue
            if candidate.direction.value == "short" and not short_context_allowed(data, timestamp, settings):
                continue
            if engine.can_open(candidate.symbol):
                engine.open_position(candidate, plan, timestamp)
                signal_count += 1

    if settings.flat_at_end:
        engine.close_all(btc_today_times[-1], last_marks, reason="session_end")

    summary = engine.summary(last_marks)
    return ReplayResult(settings=settings, engine=engine, signal_count=signal_count, marks=last_marks, summary=summary)


def optimize_winrate(
    data: dict[str, SymbolData],
    today_start_utc: datetime,
    now_utc: datetime,
    base: ReplaySettings,
    min_trades: int,
) -> dict[str, object]:
    strategy_sets = (
        frozenset(),
        frozenset({"momentum_breakout", "liquidity_sweep"}),
        frozenset({"momentum_breakout"}),
        frozenset({"liquidity_sweep"}),
        frozenset({"trend_continuation"}),
    )
    min_scores = (85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0)
    max_positions = (1, 2, 3, 5)
    ranked: list[ReplayResult] = []
    for strategies in strategy_sets:
        for min_score in min_scores:
            for max_position_count in max_positions:
                settings = ReplaySettings(
                    equity_usdt=base.equity_usdt,
                    risk_pct=base.risk_pct,
                    max_risk_pct=base.max_risk_pct,
                    volatility_adjust_risk=base.volatility_adjust_risk,
                    min_score=min_score,
                    min_volume=base.min_volume,
                    fee_bps=base.fee_bps,
                    slippage_bps=base.slippage_bps,
                    max_positions=max_position_count,
                    flat_at_end=True,
                    strategies=strategies,
                    directions=base.directions,
                    min_minutes_before_close=base.min_minutes_before_close,
                    min_minutes_after_open=base.min_minutes_after_open,
                    min_btc_24h_return_pct=base.min_btc_24h_return_pct,
                    min_btc_21d_return_pct=base.min_btc_21d_return_pct,
                    max_btc_24h_return_pct_for_shorts=base.max_btc_24h_return_pct_for_shorts,
                    max_btc_21d_return_pct_for_shorts=base.max_btc_21d_return_pct_for_shorts,
                    max_btc_72h_return_pct=base.max_btc_72h_return_pct,
                    max_btc_72h_return_pct_for_shorts=base.max_btc_72h_return_pct_for_shorts,
                    max_btc_4h_ema_gap_pct=base.max_btc_4h_ema_gap_pct,
                    max_symbol_24h_return_pct=base.max_symbol_24h_return_pct,
                    min_symbol_atr_pct_15m=base.min_symbol_atr_pct_15m,
                    breakeven_trigger_r=base.breakeven_trigger_r,
                    soft_stop_r=base.soft_stop_r,
                    max_trades=base.max_trades,
                )
                replay = run_replay(data, today_start_utc, now_utc, settings)
                if replay.summary.closed_trades >= min_trades:
                    ranked.append(replay)
    if not ranked:
        raise RuntimeError(f"no optimizer candidate produced at least {min_trades} trades")
    ranked.sort(
        key=lambda replay: (
            replay.summary.win_rate,
            replay.summary.profit_factor,
            replay.summary.total_pnl,
            -abs(replay.summary.max_drawdown_pct),
        ),
        reverse=True,
    )
    return {"best": ranked[0], "ranked": ranked[:10]}


def completed(candles: list[Candle], interval: str, now: datetime) -> list[Candle]:
    minutes = {"15m": 15, "1h": 60, "4h": 240}[interval]
    return [candle for candle in candles if candle.open_time + timedelta(minutes=minutes) <= now]


def load_timezone(name: str) -> tzinfo:
    try:
        return ZoneInfo(name)
    except Exception:
        if name in {"Asia/Calcutta", "Asia/Kolkata"}:
            return timezone(timedelta(hours=5, minutes=30), name)
        raise


def starting_equity_usdt(args: argparse.Namespace) -> float:
    if args.capital_inr is None:
        return args.equity
    if args.inr_per_usdt <= 0:
        raise ValueError("--inr-per-usdt must be positive")
    return args.capital_inr / args.inr_per_usdt


def parse_strategy_set(raw: str) -> frozenset[str]:
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def parse_filter_set(raw: str) -> frozenset[str]:
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def strategy_label(strategies: frozenset[str]) -> str:
    return ",".join(sorted(strategies)) if strategies else "all"


def index_at_or_before(candles: list[Candle], timestamp: datetime) -> int | None:
    low = 0
    high = len(candles)
    while low < high:
        mid = (low + high) // 2
        if candles[mid].open_time <= timestamp:
            low = mid + 1
        else:
            high = mid
    idx = low - 1
    return idx if idx >= 0 else None


def candles_at_or_before(candles: list[Candle], timestamp: datetime) -> list[Candle]:
    idx = index_at_or_before(candles, timestamp)
    return candles[: idx + 1] if idx is not None else []


def prior_return_pct(candles: list[Candle], timestamp: datetime, lookback_bars: int) -> float:
    idx = index_at_or_before(candles, timestamp)
    if idx is None or idx < lookback_bars:
        return 0.0
    start = candles[idx - lookback_bars].close
    end = candles[idx].close
    return (end - start) / start * 100 if start else 0.0


def long_context_allowed(
    data: dict[str, SymbolData],
    symbol_data: SymbolData,
    timestamp: datetime,
    settings: ReplaySettings,
) -> bool:
    btc_15m = data["BTCUSDT"].candles_15m
    if (
        settings.skip_long_if_btc21_min is not None
        and settings.skip_long_if_btc72_max is not None
        and settings.skip_long_if_symbol24_min is not None
        and prior_return_pct(btc_15m, timestamp, 96 * 21) >= settings.skip_long_if_btc21_min
        and prior_return_pct(btc_15m, timestamp, 288) <= settings.skip_long_if_btc72_max
        and prior_return_pct(symbol_data.candles_15m, timestamp, 96) >= settings.skip_long_if_symbol24_min
    ):
        return False
    if settings.min_btc_24h_return_pct is not None and prior_return_pct(btc_15m, timestamp, 96) < settings.min_btc_24h_return_pct:
        return False
    if settings.min_btc_21d_return_pct is not None and prior_return_pct(btc_15m, timestamp, 96 * 21) < settings.min_btc_21d_return_pct:
        return False
    if settings.max_btc_72h_return_pct is not None and prior_return_pct(btc_15m, timestamp, 288) > settings.max_btc_72h_return_pct:
        return False
    if settings.max_btc_4h_ema_gap_pct is not None and btc_4h_ema_gap_pct(data["BTCUSDT"].candles_4h, timestamp) > settings.max_btc_4h_ema_gap_pct:
        return False
    if settings.max_symbol_24h_return_pct is not None and prior_return_pct(symbol_data.candles_15m, timestamp, 96) > settings.max_symbol_24h_return_pct:
        return False
    if settings.min_symbol_atr_pct_15m is not None and symbol_atr_pct(symbol_data.candles_15m, timestamp) < settings.min_symbol_atr_pct_15m:
        return False
    return True


def short_context_allowed(
    data: dict[str, SymbolData],
    timestamp: datetime,
    settings: ReplaySettings,
) -> bool:
    if (
        settings.max_btc_24h_return_pct_for_shorts is not None
        and prior_return_pct(data["BTCUSDT"].candles_15m, timestamp, 96) > settings.max_btc_24h_return_pct_for_shorts
    ):
        return False
    if (
        settings.max_btc_21d_return_pct_for_shorts is not None
        and prior_return_pct(data["BTCUSDT"].candles_15m, timestamp, 96 * 21) > settings.max_btc_21d_return_pct_for_shorts
    ):
        return False
    if (
        settings.max_btc_72h_return_pct_for_shorts is not None
        and prior_return_pct(data["BTCUSDT"].candles_15m, timestamp, 288) > settings.max_btc_72h_return_pct_for_shorts
    ):
        return False
    return True


def btc_4h_ema_gap_pct(candles: list[Candle], timestamp: datetime) -> float:
    window = candles_at_or_before(candles, timestamp)
    if len(window) < 50:
        return 0.0
    close = window[-1].close
    if close <= 0:
        return 0.0
    return (ema([candle.close for candle in window], 20) - ema([candle.close for candle in window], 50)) / close * 100


def symbol_atr_pct(candles: list[Candle], timestamp: datetime) -> float:
    window = candles_at_or_before(candles, timestamp)
    if len(window) < 20 or window[-1].close <= 0:
        return 0.0
    return atr(frame_from_candles(window)) / window[-1].close * 100


def bar_imbalance(candle: Candle) -> float:
    span = max(candle.high - candle.low, 1e-9)
    return max(-1.0, min(1.0, (candle.close - candle.open) / span))


def session_liquidity(quote_volume: float) -> float:
    return max(0.45, min(1.0, quote_volume / 1_000_000_000))


def btc_correlation(symbol: str, candles: list[Candle], btc_candles: list[Candle]) -> float:
    if symbol == "BTCUSDT":
        return 1.0
    btc_by_time = {candle.open_time: candle.close for candle in btc_candles}
    paired: list[tuple[float, float]] = []
    for candle in candles[-50:]:
        btc_close = btc_by_time.get(candle.open_time)
        if btc_close is not None:
            paired.append((candle.close, btc_close))
    if len(paired) < 12:
        return 0.0
    asset_returns = returns([asset for asset, _ in paired])
    btc_returns = returns([btc for _, btc in paired])
    return pearson(asset_returns, btc_returns)


def returns(values: list[float]) -> list[float]:
    return [(values[idx] - values[idx - 1]) / values[idx - 1] for idx in range(1, len(values)) if values[idx - 1]]


def pearson(left: list[float], right: list[float]) -> float:
    count = min(len(left), len(right))
    if count < 2:
        return 0.0
    left = left[-count:]
    right = right[-count:]
    left_mean = sum(left) / count
    right_mean = sum(right) / count
    numerator = sum((left[idx] - left_mean) * (right[idx] - right_mean) for idx in range(count))
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    denom = sqrt(left_var * right_var)
    return numerator / denom if denom else 0.0


def write_trades(path: Path, engine: PaperTradingEngine) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol",
                "strategy",
                "direction",
                "score",
                "opened_at",
                "closed_at",
                "entry",
                "exit_price",
                "quantity",
                "realized_pnl",
                "fees_paid",
                "exit_reason",
            ]
        )
        for trade in engine.trades:
            writer.writerow(
                [
                    trade.symbol,
                    trade.strategy,
                    trade.direction,
                    f"{trade.score:.2f}",
                    trade.opened_at.isoformat(),
                    trade.closed_at.isoformat(),
                    f"{trade.entry:.8f}",
                    f"{trade.exit_price:.8f}",
                    f"{trade.quantity:.8f}",
                    f"{trade.realized_pnl:.2f}",
                    f"{trade.fees_paid:.2f}",
                    trade.exit_reason,
                ]
            )


def print_optimizer_table(ranked: list[ReplayResult]) -> None:
    print()
    print("Top win-rate filter candidates:")
    for idx, replay in enumerate(ranked, start=1):
        summary = replay.summary
        pf = "inf" if summary.profit_factor == float("inf") else f"{summary.profit_factor:.2f}"
        print(
            f"{idx:>2}. win={summary.win_rate:>5.1f}% "
            f"trades={summary.closed_trades:>2} "
            f"pnl={summary.total_pnl:>8.2f} "
            f"pf={pf:>5} "
            f"score>={replay.settings.min_score:>4.1f} "
            f"maxpos={replay.settings.max_positions} "
            f"strategies={strategy_label(replay.settings.strategies)}"
        )


def print_report(
    summary,
    engine: PaperTradingEngine,
    signal_count: int,
    day_start: datetime,
    output_path: Path,
    inr_per_usdt: float,
) -> None:
    pf = "inf" if summary.profit_factor == float("inf") else f"{summary.profit_factor:.2f}"
    print()
    print(f"Paper PnL for {day_start.date()} ({day_start.tzinfo})")
    print(f"Signals opened:        {signal_count}")
    print(f"Closed trades:         {summary.closed_trades}")
    print(f"Open positions:        {summary.open_positions}")
    print(f"Starting equity:       {summary.starting_equity:,.2f} USDT")
    print(f"Ending equity:         {summary.equity:,.2f} USDT")
    print(f"Realized PnL:          {summary.realized_pnl:,.2f} USDT")
    print(f"Unrealized PnL:        {summary.unrealized_pnl:,.2f} USDT")
    print(f"Total PnL:             {summary.total_pnl:,.2f} USDT ({summary.total_pnl_pct:.3f}%)")
    print(f"Total PnL INR est.:    {summary.total_pnl * inr_per_usdt:,.2f} INR")
    print(f"Win rate:              {summary.win_rate:.1f}%")
    print(f"Profit factor:         {pf}")
    print(f"Max realized DD:       {summary.max_drawdown_pct:.3f}%")
    print(f"Trade CSV:             {output_path}")
    if engine.trades:
        print()
        print("Closed trade sample:")
        for trade in engine.trades[:5]:
            print(
                f"  {trade.symbol} {trade.direction} {trade.strategy} "
                f"score={trade.score:.1f} pnl={trade.realized_pnl:.2f} reason={trade.exit_reason}"
            )


if __name__ == "__main__":
    main()
