from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScannerConfig:
    min_daily_volume_usd: float = 50_000_000
    max_spread_bps: float = 8.0
    min_score_to_trade: float = 85.0
    min_exchange_health_score: float = 0.8
    min_session_liquidity_score: float = 0.45
    max_news_risk_score: float = 0.75
    risk_per_trade_pct: float = 0.75
    max_risk_per_trade_pct: float = 1.0
    min_risk_per_trade_pct: float = 0.25
    volatility_adjust_risk: bool = True
