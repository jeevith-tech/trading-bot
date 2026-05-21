from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class Regime(StrEnum):
    TRENDING = "trending"
    RANGE_BOUND = "range_bound"
    VOLATILE_BREAKOUT = "volatile_breakout"
    COMPRESSION = "compression"
    PANIC_SELLOFF = "panic_selloff"
    EUPHORIA = "euphoria"
    NEWS_DRIVEN = "news_driven"
    CHOPPY = "choppy"


class StrategyKind(StrEnum):
    MOMENTUM_BREAKOUT = "momentum_breakout"
    MEAN_REVERSION = "mean_reversion"
    TREND_CONTINUATION = "trend_continuation"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    NO_TRADE = "no_trade"


class Direction(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    price: float
    daily_volume_usd: float
    spread_bps: float
    market_cap_usd: float | None = None
    funding_rate: float = 0.0
    open_interest_change_pct: float = 0.0
    btc_correlation: float = 0.0
    order_book_imbalance: float = 0.0
    whale_flow_score: float = 0.0
    stablecoin_flow_score: float = 0.0
    fear_greed: float | None = None
    session_liquidity_score: float = 1.0
    exchange_health_score: float = 1.0
    news_risk_score: float = 0.0
    candles: Mapping[str, "TimeframeFrame"] = field(default_factory=dict)


@dataclass(frozen=True)
class TimeframeFrame:
    open: tuple[float, ...]
    high: tuple[float, ...]
    low: tuple[float, ...]
    close: tuple[float, ...]
    volume: tuple[float, ...]

    def __post_init__(self) -> None:
        lengths = {len(self.open), len(self.high), len(self.low), len(self.close), len(self.volume)}
        if len(lengths) != 1:
            raise ValueError("OHLCV arrays must have the same length")
        if not self.close:
            raise ValueError("timeframe frame must contain at least one candle")


@dataclass(frozen=True)
class RiskState:
    equity: float
    daily_pnl_pct: float = 0.0
    weekly_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    open_risk_pct: float = 0.0
    max_open_risk_pct: float = 3.0


@dataclass(frozen=True)
class ScoreBreakdown:
    htf_trend_alignment: float
    volume_strength: float
    market_structure: float
    btc_correlation: float
    liquidity_conditions: float
    open_interest: float
    volatility_conditions: float
    entry_timing: float

    @property
    def total(self) -> float:
        return round(
            self.htf_trend_alignment
            + self.volume_strength
            + self.market_structure
            + self.btc_correlation
            + self.liquidity_conditions
            + self.open_interest
            + self.volatility_conditions
            + self.entry_timing,
            2,
        )


@dataclass(frozen=True)
class SetupCandidate:
    symbol: str
    strategy: StrategyKind
    direction: Direction
    score: float
    regime: Regime
    entry: float
    stop: float
    target_1: float
    target_2: float
    reasons: tuple[str, ...]
    score_breakdown: ScoreBreakdown
    risk_rejected: bool = False


@dataclass(frozen=True)
class PositionPlan:
    symbol: str
    direction: Direction
    entry: float
    stop: float
    stop_distance: float
    account_risk: float
    quantity: float
    notional: float
    risk_pct: float
    tp1: float
    tp2: float
    runner_enabled: bool
