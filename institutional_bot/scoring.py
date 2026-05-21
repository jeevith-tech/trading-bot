from __future__ import annotations

from .indicators import ema, slope, volume_expansion
from .models import Direction, MarketSnapshot, Regime, ScoreBreakdown, StrategyKind


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_setup(
    snapshot: MarketSnapshot,
    strategy: StrategyKind,
    direction: Direction,
    regime: Regime,
    reasons: tuple[str, ...],
) -> ScoreBreakdown:
    frame_15m = snapshot.candles.get("15m") or snapshot.candles.get("5m")
    frame_4h = snapshot.candles.get("4h") or snapshot.candles.get("1h") or frame_15m
    if frame_15m is None or frame_4h is None:
        return ScoreBreakdown(0, 0, 0, 0, 0, 0, 0, 0)

    htf_slope = slope(frame_4h.close, 30)
    fast = ema(frame_4h.close, 20)
    slow = ema(frame_4h.close, 50)
    aligned_long = direction == Direction.LONG and htf_slope > 0 and fast > slow
    aligned_short = direction == Direction.SHORT and htf_slope < 0 and fast < slow
    trend_alignment = 20 if aligned_long or aligned_short else clamp(8 - abs(htf_slope) * 100, 0, 12)

    vol = clamp((volume_expansion(frame_15m) - 1) / 1.6, 0, 1) * 15
    structure_bonus = min(15, 6 + len(reasons) * 2.5)
    btc_supportive = (
        direction == Direction.LONG and snapshot.btc_correlation >= -0.2
    ) or (
        direction == Direction.SHORT and snapshot.btc_correlation <= 0.85
    )
    btc_score = 10 if btc_supportive else 4
    liquidity = clamp(snapshot.session_liquidity_score, 0, 1) * 5
    liquidity += clamp(snapshot.exchange_health_score, 0, 1) * 3
    liquidity += clamp(1 - snapshot.spread_bps / 10, 0, 1) * 2
    oi = clamp(abs(snapshot.open_interest_change_pct) / 8, 0, 1) * 10

    volatility = {
        Regime.TRENDING: 9,
        Regime.VOLATILE_BREAKOUT: 8,
        Regime.COMPRESSION: 7,
        Regime.RANGE_BOUND: 8 if strategy == StrategyKind.MEAN_REVERSION else 4,
        Regime.PANIC_SELLOFF: 5,
        Regime.EUPHORIA: 5,
        Regime.NEWS_DRIVEN: 1,
        Regime.CHOPPY: 2,
    }[regime]
    timing = 7 + min(3, max(0, snapshot.order_book_imbalance * 3 if direction == Direction.LONG else -snapshot.order_book_imbalance * 3))

    return ScoreBreakdown(
        htf_trend_alignment=round(trend_alignment, 2),
        volume_strength=round(vol, 2),
        market_structure=round(structure_bonus, 2),
        btc_correlation=round(btc_score, 2),
        liquidity_conditions=round(liquidity, 2),
        open_interest=round(oi, 2),
        volatility_conditions=round(volatility, 2),
        entry_timing=round(timing, 2),
    )
