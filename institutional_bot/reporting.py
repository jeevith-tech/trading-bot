from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .models import PositionPlan, SetupCandidate


@dataclass(frozen=True)
class TradeLog:
    timestamp: datetime
    candidate: SetupCandidate
    position: PositionPlan
    entry_reason: str
    exit_reason: str | None = None
    market_emotion: str = "neutral"
    screenshot_path: str | None = None
    result_r: float | None = None

    @classmethod
    def from_candidate(cls, candidate: SetupCandidate, position: PositionPlan) -> "TradeLog":
        return cls(
            timestamp=datetime.now(timezone.utc),
            candidate=candidate,
            position=position,
            entry_reason="; ".join(candidate.reasons),
            market_emotion=_market_emotion(candidate),
        )


def _market_emotion(candidate: SetupCandidate) -> str:
    if candidate.regime.value in {"panic_selloff", "euphoria", "news_driven"}:
        return candidate.regime.value
    if candidate.score >= 92:
        return "high-conviction but controlled"
    return "selective"
