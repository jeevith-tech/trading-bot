from __future__ import annotations

from .indicators import atr, bollinger_width, ema, rsi, slope, volume_expansion, vwap
from .models import Direction, MarketSnapshot, Regime, SetupCandidate, StrategyKind
from .scoring import score_setup


def detect_candidates(snapshot: MarketSnapshot, regime: Regime) -> list[SetupCandidate]:
    detectors = (
        _momentum_breakout,
        _trend_continuation,
        _mean_reversion,
        _liquidity_sweep,
    )
    candidates: list[SetupCandidate] = []
    for detector in detectors:
        candidate = detector(snapshot, regime)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _build_candidate(
    snapshot: MarketSnapshot,
    strategy: StrategyKind,
    direction: Direction,
    regime: Regime,
    entry: float,
    stop: float,
    reasons: tuple[str, ...],
) -> SetupCandidate:
    risk = abs(entry - stop)
    target_1 = entry + risk if direction == Direction.LONG else entry - risk
    target_2 = entry + 2 * risk if direction == Direction.LONG else entry - 2 * risk
    breakdown = score_setup(snapshot, strategy, direction, regime, reasons)
    return SetupCandidate(
        symbol=snapshot.symbol,
        strategy=strategy,
        direction=direction,
        score=breakdown.total,
        regime=regime,
        entry=entry,
        stop=stop,
        target_1=target_1,
        target_2=target_2,
        reasons=reasons,
        score_breakdown=breakdown,
    )


def _momentum_breakout(snapshot: MarketSnapshot, regime: Regime) -> SetupCandidate | None:
    frame = snapshot.candles.get("15m") or snapshot.candles.get("5m")
    htf = snapshot.candles.get("4h") or snapshot.candles.get("1h")
    if frame is None or htf is None or len(frame.close) < 30:
        return None

    last = frame.close[-1]
    recent_high = max(frame.high[-21:-1])
    recent_low = min(frame.low[-21:-1])
    compression = bollinger_width(frame.close[:-1]) < 0.055
    expansion = volume_expansion(frame) >= 2.0
    htf_up = ema(htf.close, 20) > ema(htf.close, 50) and slope(htf.close, 30) > 0
    htf_down = ema(htf.close, 20) < ema(htf.close, 50) and slope(htf.close, 30) < 0
    strong_close_up = last > recent_high and frame.close[-1] > (frame.high[-1] + frame.low[-1]) / 2
    strong_close_down = last < recent_low and frame.close[-1] < (frame.high[-1] + frame.low[-1]) / 2
    oi_support = snapshot.open_interest_change_pct > 1.5

    if compression and expansion and oi_support and htf_up and strong_close_up:
        stop = min(recent_high, last - 1.5 * atr(frame))
        return _build_candidate(
            snapshot,
            StrategyKind.MOMENTUM_BREAKOUT,
            Direction.LONG,
            regime,
            last,
            stop,
            ("volatility compression", "2x volume expansion", "open interest increasing", "HTF trend bullish", "breakout close"),
        )
    if compression and expansion and oi_support and htf_down and strong_close_down:
        stop = max(recent_low, last + 1.5 * atr(frame))
        return _build_candidate(
            snapshot,
            StrategyKind.MOMENTUM_BREAKOUT,
            Direction.SHORT,
            regime,
            last,
            stop,
            ("volatility compression", "2x volume expansion", "open interest increasing", "HTF trend bearish", "breakdown close"),
        )
    return None


def _trend_continuation(snapshot: MarketSnapshot, regime: Regime) -> SetupCandidate | None:
    frame = snapshot.candles.get("15m") or snapshot.candles.get("5m")
    htf = snapshot.candles.get("4h") or snapshot.candles.get("1h")
    if frame is None or htf is None or regime not in {Regime.TRENDING, Regime.VOLATILE_BREAKOUT}:
        return None

    last = frame.close[-1]
    ema20 = ema(frame.close, 20)
    ema50 = ema(frame.close, 50)
    avg_pullback_volume = sum(frame.volume[-6:-1]) / max(len(frame.volume[-6:-1]), 1)
    impulse = frame.volume[-1] > avg_pullback_volume * 1.3
    htf_up = ema(htf.close, 20) > ema(htf.close, 50) and slope(htf.close, 30) > 0.02
    htf_down = ema(htf.close, 20) < ema(htf.close, 50) and slope(htf.close, 30) < -0.02

    if htf_up and min(ema20, ema50) <= last <= max(ema20, ema50) * 1.018 and impulse and last > frame.open[-1]:
        stop = min(frame.low[-8:]) - 0.25 * atr(frame)
        return _build_candidate(
            snapshot,
            StrategyKind.TREND_CONTINUATION,
            Direction.LONG,
            regime,
            last,
            stop,
            ("strong HTF trend", "pullback into EMA cluster", "pullback volume fading", "impulse resumption"),
        )
    if htf_down and min(ema20, ema50) * 0.982 <= last <= max(ema20, ema50) and impulse and last < frame.open[-1]:
        stop = max(frame.high[-8:]) + 0.25 * atr(frame)
        return _build_candidate(
            snapshot,
            StrategyKind.TREND_CONTINUATION,
            Direction.SHORT,
            regime,
            last,
            stop,
            ("strong HTF trend", "pullback into EMA cluster", "pullback volume fading", "impulse resumption"),
        )
    return None


def _mean_reversion(snapshot: MarketSnapshot, regime: Regime) -> SetupCandidate | None:
    frame = snapshot.candles.get("15m") or snapshot.candles.get("5m")
    if frame is None or regime not in {Regime.RANGE_BOUND, Regime.PANIC_SELLOFF, Regime.EUPHORIA}:
        return None

    last = frame.close[-1]
    fair = vwap(frame)
    deviation = (last - fair) / max(fair, 1e-9)
    exhaustion = rsi(frame.close)
    wick_down = (min(frame.open[-1], frame.close[-1]) - frame.low[-1]) / max(last, 1e-9)
    wick_up = (frame.high[-1] - max(frame.open[-1], frame.close[-1])) / max(last, 1e-9)
    cascade = abs(snapshot.funding_rate) > 0.0008 or abs(snapshot.open_interest_change_pct) > 5

    if deviation < -0.025 and exhaustion < 28 and wick_down > 0.008 and cascade:
        stop = frame.low[-1] - 0.5 * atr(frame)
        return _build_candidate(
            snapshot,
            StrategyKind.MEAN_REVERSION,
            Direction.LONG,
            regime,
            last,
            stop,
            ("extreme VWAP deviation", "RSI exhaustion", "liquidation-style wick", "biased funding or OI flush"),
        )
    if deviation > 0.025 and exhaustion > 72 and wick_up > 0.008 and cascade:
        stop = frame.high[-1] + 0.5 * atr(frame)
        return _build_candidate(
            snapshot,
            StrategyKind.MEAN_REVERSION,
            Direction.SHORT,
            regime,
            last,
            stop,
            ("extreme VWAP deviation", "RSI exhaustion", "liquidation-style wick", "biased funding or OI flush"),
        )
    return None


def _liquidity_sweep(snapshot: MarketSnapshot, regime: Regime) -> SetupCandidate | None:
    frame = snapshot.candles.get("5m") or snapshot.candles.get("15m")
    if frame is None or len(frame.close) < 20:
        return None

    last = frame.close[-1]
    prior_high = max(frame.high[-16:-1])
    prior_low = min(frame.low[-16:-1])
    swept_high = frame.high[-1] > prior_high and last < prior_high
    swept_low = frame.low[-1] < prior_low and last > prior_low
    oi_flush = snapshot.open_interest_change_pct < -2.0
    delta_reversal_long = snapshot.order_book_imbalance > 0.2
    delta_reversal_short = snapshot.order_book_imbalance < -0.2

    if swept_low and oi_flush and delta_reversal_long:
        stop = frame.low[-1] - 0.25 * atr(frame)
        return _build_candidate(
            snapshot,
            StrategyKind.LIQUIDITY_SWEEP,
            Direction.LONG,
            regime,
            last,
            stop,
            ("sell-side liquidity sweep", "immediate rejection", "order-book reversal", "open interest flush"),
        )
    if swept_high and oi_flush and delta_reversal_short:
        stop = frame.high[-1] + 0.25 * atr(frame)
        return _build_candidate(
            snapshot,
            StrategyKind.LIQUIDITY_SWEEP,
            Direction.SHORT,
            regime,
            last,
            stop,
            ("buy-side liquidity sweep", "immediate rejection", "order-book reversal", "open interest flush"),
        )
    return None
