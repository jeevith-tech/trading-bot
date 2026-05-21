from __future__ import annotations

from .config import ScannerConfig
from .models import Direction, PositionPlan, RiskState, SetupCandidate


def trading_allowed(risk_state: RiskState) -> tuple[bool, str | None]:
    if risk_state.daily_pnl_pct <= -3.0:
        return False, "daily loss limit reached"
    if risk_state.weekly_drawdown_pct <= -8.0:
        return False, "weekly drawdown limit reached"
    if risk_state.consecutive_losses >= 3:
        return False, "three consecutive losses"
    if risk_state.open_risk_pct >= risk_state.max_open_risk_pct:
        return False, "maximum open risk reached"
    return True, None


def volatility_adjusted_risk_pct(
    base_risk_pct: float,
    atr_pct: float,
    config: ScannerConfig,
) -> float:
    risk = base_risk_pct
    if atr_pct > 0.045:
        risk *= 0.45
    elif atr_pct > 0.03:
        risk *= 0.65
    elif atr_pct < 0.012:
        risk *= 0.85
    return max(config.min_risk_per_trade_pct, min(config.max_risk_per_trade_pct, risk))


def build_position_plan(
    candidate: SetupCandidate,
    risk_state: RiskState,
    risk_pct: float,
) -> PositionPlan:
    if candidate.direction == Direction.FLAT:
        raise ValueError("cannot size a flat candidate")
    stop_distance = abs(candidate.entry - candidate.stop)
    if stop_distance <= 0:
        raise ValueError("stop distance must be positive")
    account_risk = risk_state.equity * (risk_pct / 100)
    quantity = account_risk / stop_distance
    return PositionPlan(
        symbol=candidate.symbol,
        direction=candidate.direction,
        entry=candidate.entry,
        stop=candidate.stop,
        stop_distance=stop_distance,
        account_risk=account_risk,
        quantity=quantity,
        notional=quantity * candidate.entry,
        risk_pct=risk_pct,
        tp1=candidate.target_1,
        tp2=candidate.target_2,
        runner_enabled=True,
    )
