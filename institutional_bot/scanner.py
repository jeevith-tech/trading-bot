from __future__ import annotations

from dataclasses import dataclass

from .config import ScannerConfig
from .indicators import atr
from .models import MarketSnapshot, PositionPlan, RiskState, SetupCandidate
from .regime import classify_regime
from .risk import build_position_plan, trading_allowed, volatility_adjusted_risk_pct
from .strategies import detect_candidates


@dataclass(frozen=True)
class ScanDecision:
    tradable: list[tuple[SetupCandidate, PositionPlan]]
    rejected: list[tuple[SetupCandidate | None, str]]


class MarketScanner:
    def __init__(self, config: ScannerConfig | None = None) -> None:
        self.config = config or ScannerConfig()

    def scan(self, snapshots: list[MarketSnapshot], risk_state: RiskState) -> ScanDecision:
        rejected: list[tuple[SetupCandidate | None, str]] = []
        tradable: list[tuple[SetupCandidate, PositionPlan]] = []

        allowed, reason = trading_allowed(risk_state)
        if not allowed:
            return ScanDecision(tradable=[], rejected=[(None, reason or "risk gate closed")])

        for snapshot in snapshots:
            market_rejection = self._market_filter(snapshot)
            if market_rejection:
                rejected.append((None, f"{snapshot.symbol}: {market_rejection}"))
                continue

            regime = classify_regime(snapshot)
            for candidate in detect_candidates(snapshot, regime):
                if candidate.score < self.config.min_score_to_trade:
                    rejected.append((candidate, "score below A+ threshold"))
                    continue
                frame = snapshot.candles.get("15m") or snapshot.candles.get("5m")
                atr_pct = atr(frame) / snapshot.price if frame else 0.02
                if self.config.volatility_adjust_risk:
                    risk_pct = volatility_adjusted_risk_pct(self.config.risk_per_trade_pct, atr_pct, self.config)
                else:
                    risk_pct = min(self.config.max_risk_per_trade_pct, self.config.risk_per_trade_pct)
                plan = build_position_plan(candidate, risk_state, risk_pct)
                tradable.append((candidate, plan))

        tradable.sort(key=lambda item: item[0].score, reverse=True)
        return ScanDecision(tradable=tradable, rejected=rejected)

    def _market_filter(self, snapshot: MarketSnapshot) -> str | None:
        if snapshot.daily_volume_usd < self.config.min_daily_volume_usd:
            return "daily volume below liquidity threshold"
        if snapshot.spread_bps > self.config.max_spread_bps:
            return "spread too wide"
        if snapshot.exchange_health_score < self.config.min_exchange_health_score:
            return "exchange health degraded"
        if snapshot.session_liquidity_score < self.config.min_session_liquidity_score:
            return "session liquidity too low"
        if snapshot.news_risk_score > self.config.max_news_risk_score:
            return "news volatility risk too high"
        return None
