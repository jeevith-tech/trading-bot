from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .models import Direction, PositionPlan, RiskState, SetupCandidate


@dataclass(frozen=True)
class PaperConfig:
    starting_equity: float = 100_000.0
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    max_concurrent_positions: int = 5
    tp1_fraction: float = 0.50
    tp2_fraction: float = 0.30
    move_stop_to_breakeven_after_tp1: bool = True
    breakeven_trigger_r: float | None = None
    soft_stop_r: float | None = None


@dataclass
class PaperPosition:
    candidate: SetupCandidate
    plan: PositionPlan
    opened_at: datetime
    entry: float
    quantity: float
    stop: float
    tp1: float
    tp2: float
    initial_risk_distance: float
    remaining_quantity: float
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    closed_at: datetime | None = None
    exit_reason: str | None = None

    @property
    def is_open(self) -> bool:
        return self.remaining_quantity > 1e-12 and self.closed_at is None


@dataclass(frozen=True)
class PaperTrade:
    symbol: str
    strategy: str
    direction: str
    score: float
    opened_at: datetime
    closed_at: datetime
    entry: float
    exit_price: float
    quantity: float
    realized_pnl: float
    fees_paid: float
    exit_reason: str


@dataclass(frozen=True)
class PaperSummary:
    starting_equity: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    total_pnl_pct: float
    closed_trades: int
    open_positions: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float


@dataclass
class PaperTradingEngine:
    config: PaperConfig = field(default_factory=PaperConfig)
    positions: list[PaperPosition] = field(default_factory=list)
    trades: list[PaperTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.equity_curve.append(self.config.starting_equity)

    @property
    def realized_pnl(self) -> float:
        return sum(position.realized_pnl for position in self.positions)

    @property
    def equity(self) -> float:
        return self.config.starting_equity + self.realized_pnl

    def risk_state(self) -> RiskState:
        pnl_pct = (self.equity - self.config.starting_equity) / self.config.starting_equity * 100
        return RiskState(
            equity=self.equity,
            daily_pnl_pct=pnl_pct,
            weekly_drawdown_pct=0.0,
            consecutive_losses=self._consecutive_losses(),
            open_risk_pct=self._open_risk_pct(),
        )

    def can_open(self, symbol: str) -> bool:
        open_positions = [position for position in self.positions if position.is_open]
        if len(open_positions) >= self.config.max_concurrent_positions:
            return False
        return all(position.candidate.symbol != symbol for position in open_positions)

    def open_position(self, candidate: SetupCandidate, plan: PositionPlan, timestamp: datetime) -> PaperPosition:
        if candidate.direction == Direction.LONG:
            fill = candidate.entry * (1 + self.config.slippage_bps / 10_000)
        elif candidate.direction == Direction.SHORT:
            fill = candidate.entry * (1 - self.config.slippage_bps / 10_000)
        else:
            raise ValueError("cannot open a flat paper position")

        fee = abs(fill * plan.quantity) * self.config.fee_bps / 10_000
        position = PaperPosition(
            candidate=candidate,
            plan=plan,
            opened_at=timestamp,
            entry=fill,
            quantity=plan.quantity,
            stop=plan.stop,
            tp1=plan.tp1,
            tp2=plan.tp2,
            initial_risk_distance=abs(fill - plan.stop),
            remaining_quantity=plan.quantity,
            realized_pnl=-fee,
            fees_paid=fee,
        )
        self.positions.append(position)
        self._record_equity()
        return position

    def update_bar(self, symbol: str, timestamp: datetime, high: float, low: float, close: float) -> None:
        for position in list(self.positions):
            if not position.is_open or position.candidate.symbol != symbol:
                continue
            self._manage_position(position, timestamp, high, low, close)
        self._record_equity()

    def close_all(self, timestamp: datetime, marks: dict[str, float], reason: str = "mark_to_market") -> None:
        for position in list(self.positions):
            if not position.is_open:
                continue
            mark = marks.get(position.candidate.symbol)
            if mark is None:
                continue
            self._close_fraction(position, timestamp, mark, 1.0, reason)
        self._record_equity()

    def summary(self, marks: dict[str, float] | None = None) -> PaperSummary:
        marks = marks or {}
        unrealized = sum(self._unrealized(position, marks) for position in self.positions if position.is_open)
        total_pnl = self.realized_pnl + unrealized
        wins = sum(1 for trade in self.trades if trade.realized_pnl > 0)
        losses = sum(1 for trade in self.trades if trade.realized_pnl <= 0)
        gross_profit = sum(trade.realized_pnl for trade in self.trades if trade.realized_pnl > 0)
        gross_loss = abs(sum(trade.realized_pnl for trade in self.trades if trade.realized_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
        closed = len(self.trades)
        return PaperSummary(
            starting_equity=self.config.starting_equity,
            equity=self.config.starting_equity + total_pnl,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl / self.config.starting_equity * 100,
            closed_trades=closed,
            open_positions=sum(1 for position in self.positions if position.is_open),
            wins=wins,
            losses=losses,
            win_rate=(wins / closed * 100) if closed else 0.0,
            profit_factor=profit_factor,
            max_drawdown_pct=self._max_drawdown_pct(),
        )

    def _manage_position(
        self,
        position: PaperPosition,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
    ) -> None:
        if position.candidate.direction == Direction.LONG:
            stop_hit = low <= position.stop
            tp1_hit = high >= position.tp1
            tp2_hit = high >= position.tp2
        else:
            stop_hit = high >= position.stop
            tp1_hit = low <= position.tp1
            tp2_hit = low <= position.tp2

        if stop_hit:
            self._close_fraction(position, timestamp, position.stop, 1.0, "stop_loss")
            return
        self._move_stop_to_breakeven_if_triggered(position, high, low)
        if tp1_hit and not position.tp1_hit:
            self._close_fraction(position, timestamp, position.tp1, self.config.tp1_fraction, "tp1")
            position.tp1_hit = True
            if self.config.move_stop_to_breakeven_after_tp1:
                position.stop = position.entry
        if tp2_hit and not position.tp2_hit and position.is_open:
            self._close_fraction(position, timestamp, position.tp2, self.config.tp2_fraction, "tp2")
            position.tp2_hit = True
        if not position.is_open:
            return
        if self._soft_stop_triggered(position, close):
            self._close_fraction(position, timestamp, close, 1.0, "soft_stop")
            return
        self._mark_equity(close)

    def _move_stop_to_breakeven_if_triggered(self, position: PaperPosition, high: float, low: float) -> None:
        if self.config.breakeven_trigger_r is None or position.initial_risk_distance <= 0:
            return
        trigger = self.config.breakeven_trigger_r * position.initial_risk_distance
        if position.candidate.direction == Direction.LONG and high >= position.entry + trigger:
            position.stop = max(position.stop, position.entry)
        elif position.candidate.direction == Direction.SHORT and low <= position.entry - trigger:
            position.stop = min(position.stop, position.entry)

    def _soft_stop_triggered(self, position: PaperPosition, close: float) -> bool:
        if self.config.soft_stop_r is None or position.initial_risk_distance <= 0:
            return False
        if position.candidate.direction == Direction.LONG:
            adverse_r = (position.entry - close) / position.initial_risk_distance
        else:
            adverse_r = (close - position.entry) / position.initial_risk_distance
        return adverse_r >= self.config.soft_stop_r

    def _close_fraction(
        self,
        position: PaperPosition,
        timestamp: datetime,
        price: float,
        fraction: float,
        reason: str,
    ) -> None:
        quantity = min(position.remaining_quantity, position.quantity * fraction)
        if quantity <= 0:
            return
        if position.candidate.direction == Direction.LONG:
            fill = price * (1 - self.config.slippage_bps / 10_000)
            pnl = (fill - position.entry) * quantity
        else:
            fill = price * (1 + self.config.slippage_bps / 10_000)
            pnl = (position.entry - fill) * quantity
        fee = abs(fill * quantity) * self.config.fee_bps / 10_000
        position.remaining_quantity -= quantity
        position.realized_pnl += pnl - fee
        position.fees_paid += fee
        if position.remaining_quantity <= 1e-12:
            position.closed_at = timestamp
            position.exit_reason = reason
            self.trades.append(
                PaperTrade(
                    symbol=position.candidate.symbol,
                    strategy=position.candidate.strategy.value,
                    direction=position.candidate.direction.value,
                    score=position.candidate.score,
                    opened_at=position.opened_at,
                    closed_at=timestamp,
                    entry=position.entry,
                    exit_price=fill,
                    quantity=position.quantity,
                    realized_pnl=position.realized_pnl,
                    fees_paid=position.fees_paid,
                    exit_reason=reason,
                )
            )

    def _unrealized(self, position: PaperPosition, marks: dict[str, float]) -> float:
        mark = marks.get(position.candidate.symbol)
        if mark is None:
            return 0.0
        if position.candidate.direction == Direction.LONG:
            exit_mark = mark * (1 - self.config.slippage_bps / 10_000)
            pnl = (exit_mark - position.entry) * position.remaining_quantity
        else:
            exit_mark = mark * (1 + self.config.slippage_bps / 10_000)
            pnl = (position.entry - exit_mark) * position.remaining_quantity
        close_fee = abs(exit_mark * position.remaining_quantity) * self.config.fee_bps / 10_000
        return pnl - close_fee

    def _record_equity(self) -> None:
        self.equity_curve.append(self.equity)

    def _mark_equity(self, close: float) -> None:
        if self.equity_curve:
            self.equity_curve[-1] = self.equity
        else:
            self.equity_curve.append(self.equity)

    def _max_drawdown_pct(self) -> float:
        peak = self.config.starting_equity
        max_drawdown = 0.0
        for equity in self.equity_curve:
            peak = max(peak, equity)
            if peak:
                max_drawdown = min(max_drawdown, (equity - peak) / peak * 100)
        return max_drawdown

    def _consecutive_losses(self) -> int:
        losses = 0
        for trade in reversed(self.trades):
            if trade.realized_pnl <= 0:
                losses += 1
            else:
                break
        return losses

    def _open_risk_pct(self) -> float:
        account = max(self.equity, 1e-9)
        risk = sum(position.plan.account_risk for position in self.positions if position.is_open)
        return risk / account * 100
