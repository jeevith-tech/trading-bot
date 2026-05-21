from __future__ import annotations

from .indicators import atr, bollinger_width, ema, rsi, slope
from .models import MarketSnapshot, Regime


def classify_regime(snapshot: MarketSnapshot) -> Regime:
    frame = snapshot.candles.get("1h") or snapshot.candles.get("15m")
    htf = snapshot.candles.get("4h") or frame
    if frame is None or htf is None:
        return Regime.CHOPPY

    close = frame.close
    last = close[-1]
    atr_pct = atr(frame) / max(last, 1e-9)
    width = bollinger_width(close)
    trend = slope(htf.close, 30)
    fast = ema(htf.close, 20)
    slow = ema(htf.close, 50)
    exhaustion = rsi(close)

    if snapshot.news_risk_score >= 0.75:
        return Regime.NEWS_DRIVEN
    if atr_pct > 0.055 and trend < -0.05:
        return Regime.PANIC_SELLOFF
    if atr_pct > 0.05 and trend > 0.05 and exhaustion > 78:
        return Regime.EUPHORIA
    if width < 0.025 and atr_pct < 0.018:
        return Regime.COMPRESSION
    if atr_pct > 0.035 and abs(trend) > 0.03:
        return Regime.VOLATILE_BREAKOUT
    if abs(trend) > 0.025 and ((fast > slow and trend > 0) or (fast < slow and trend < 0)):
        return Regime.TRENDING
    if width < 0.08 and abs(trend) < 0.02:
        return Regime.RANGE_BOUND
    return Regime.CHOPPY
