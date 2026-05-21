from __future__ import annotations

import math
from statistics import mean

from .models import TimeframeFrame


def sma(values: tuple[float, ...], period: int) -> float:
    if len(values) < period:
        return mean(values)
    return mean(values[-period:])


def ema(values: tuple[float, ...], period: int) -> float:
    if not values:
        raise ValueError("ema requires values")
    alpha = 2 / (period + 1)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1 - alpha) * result
    return result


def rsi(values: tuple[float, ...], period: int = 14) -> float:
    if len(values) < 2:
        return 50.0
    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    window = changes[-period:]
    gains = [max(change, 0.0) for change in window]
    losses = [abs(min(change, 0.0)) for change in window]
    avg_gain = mean(gains) if gains else 0.0
    avg_loss = mean(losses) if losses else 0.0
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(frame: TimeframeFrame, period: int = 14) -> float:
    if len(frame.close) < 2:
        return frame.high[-1] - frame.low[-1]
    true_ranges: list[float] = []
    for idx in range(1, len(frame.close)):
        true_ranges.append(
            max(
                frame.high[idx] - frame.low[idx],
                abs(frame.high[idx] - frame.close[idx - 1]),
                abs(frame.low[idx] - frame.close[idx - 1]),
            )
        )
    return sma(tuple(true_ranges), period)


def vwap(frame: TimeframeFrame, period: int = 80) -> float:
    start = max(0, len(frame.close) - period)
    numerator = 0.0
    denominator = 0.0
    for high, low, close, volume in zip(
        frame.high[start:], frame.low[start:], frame.close[start:], frame.volume[start:]
    ):
        typical = (high + low + close) / 3
        numerator += typical * volume
        denominator += volume
    return numerator / denominator if denominator else frame.close[-1]


def bollinger_width(values: tuple[float, ...], period: int = 20) -> float:
    window = values[-period:] if len(values) >= period else values
    center = mean(window)
    if center == 0:
        return 0.0
    variance = mean([(value - center) ** 2 for value in window])
    return (4 * math.sqrt(variance)) / center


def slope(values: tuple[float, ...], period: int = 20) -> float:
    if len(values) < 2:
        return 0.0
    window = values[-period:] if len(values) >= period else values
    return (window[-1] - window[0]) / max(abs(window[0]), 1e-9)


def volume_expansion(frame: TimeframeFrame, period: int = 30) -> float:
    baseline = sma(frame.volume[:-1] or frame.volume, period)
    return frame.volume[-1] / baseline if baseline else 1.0
