from __future__ import annotations

from datetime import datetime, timezone

from institutional_bot import MarketScanner, MarketSnapshot, PaperTradingEngine, RiskState, StrategyKind
from institutional_bot.config import ScannerConfig
from institutional_bot.models import Direction, TimeframeFrame
from institutional_bot.risk import build_position_plan


def make_frame(
    start: float,
    step: float,
    count: int = 80,
    volume: float = 1_000_000,
    breakout: bool = False,
) -> TimeframeFrame:
    closes = [start + step * idx for idx in range(count)]
    opens = [close - step * 0.35 for close in closes]
    highs = [max(open_, close) * 1.002 for open_, close in zip(opens, closes)]
    lows = [min(open_, close) * 0.998 for open_, close in zip(opens, closes)]
    volumes = [volume for _ in closes]
    if breakout:
        closes[-25:-1] = [closes[-25] + ((idx % 3) - 1) * start * 0.0004 for idx in range(24)]
        opens[-25:-1] = [close * 0.9998 for close in closes[-25:-1]]
        highs[-25:-1] = [close * 1.001 for close in closes[-25:-1]]
        lows[-25:-1] = [close * 0.999 for close in closes[-25:-1]]
        closes[-1] = max(highs[-25:-1]) * 1.02
        opens[-1] = closes[-1] * 0.993
        highs[-1] = closes[-1] * 1.004
        lows[-1] = opens[-1] * 0.998
        volumes[-1] = volume * 3.2
    return TimeframeFrame(
        open=tuple(opens),
        high=tuple(highs),
        low=tuple(lows),
        close=tuple(closes),
        volume=tuple(volumes),
    )


def test_scanner_finds_only_a_plus_breakout() -> None:
    snapshot = MarketSnapshot(
        symbol="BTC/USDT",
        price=72_000,
        daily_volume_usd=20_000_000_000,
        spread_bps=1.2,
        funding_rate=0.0002,
        open_interest_change_pct=6.5,
        btc_correlation=1.0,
        order_book_imbalance=0.45,
        session_liquidity_score=0.95,
        exchange_health_score=0.99,
        candles={
            "15m": make_frame(70_000, 4, breakout=True),
            "4h": make_frame(62_000, 120),
        },
    )
    decision = MarketScanner().scan([snapshot], RiskState(equity=100_000))
    assert len(decision.tradable) == 1
    candidate, plan = decision.tradable[0]
    assert candidate.strategy == StrategyKind.MOMENTUM_BREAKOUT
    assert candidate.direction == Direction.LONG
    assert candidate.score >= 85
    assert 0 < plan.account_risk <= 1_000


def test_risk_kill_switch_blocks_trading() -> None:
    snapshot = MarketSnapshot(
        symbol="ETH/USDT",
        price=3_500,
        daily_volume_usd=5_000_000_000,
        spread_bps=1.5,
        candles={"15m": make_frame(3_400, 1, breakout=True), "4h": make_frame(3_000, 8)},
    )
    decision = MarketScanner().scan([snapshot], RiskState(equity=100_000, consecutive_losses=3))
    assert not decision.tradable
    assert decision.rejected[0][1] == "three consecutive losses"


def test_liquidity_filter_rejects_low_quality_market() -> None:
    snapshot = MarketSnapshot(
        symbol="TINY/USDT",
        price=0.05,
        daily_volume_usd=1_000_000,
        spread_bps=25,
        candles={"15m": make_frame(0.05, 0.0001), "4h": make_frame(0.04, 0.0001)},
    )
    decision = MarketScanner().scan([snapshot], RiskState(equity=100_000))
    assert not decision.tradable
    assert "daily volume below liquidity threshold" in decision.rejected[0][1]


def test_paper_engine_accounts_for_fees_and_tp() -> None:
    snapshot = MarketSnapshot(
        symbol="BTC/USDT",
        price=72_000,
        daily_volume_usd=20_000_000_000,
        spread_bps=1.2,
        funding_rate=0.0002,
        open_interest_change_pct=6.5,
        btc_correlation=1.0,
        order_book_imbalance=0.45,
        session_liquidity_score=0.95,
        exchange_health_score=0.99,
        candles={
            "15m": make_frame(70_000, 4, breakout=True),
            "4h": make_frame(62_000, 120),
        },
    )
    candidate, plan = MarketScanner().scan([snapshot], RiskState(equity=100_000)).tradable[0]
    engine = PaperTradingEngine()
    engine.open_position(candidate, plan, datetime.now(timezone.utc))
    engine.update_bar(
        candidate.symbol,
        datetime.now(timezone.utc),
        high=candidate.target_2 * 1.01,
        low=candidate.entry * 0.999,
        close=candidate.target_2,
    )
    summary = engine.summary({candidate.symbol: candidate.target_2})
    assert summary.total_pnl > 0
    assert summary.open_positions == 1


def test_position_sizing_compounds_from_current_equity() -> None:
    snapshot = MarketSnapshot(
        symbol="BTC/USDT",
        price=72_000,
        daily_volume_usd=20_000_000_000,
        spread_bps=1.2,
        funding_rate=0.0002,
        open_interest_change_pct=6.5,
        btc_correlation=1.0,
        order_book_imbalance=0.45,
        session_liquidity_score=0.95,
        exchange_health_score=0.99,
        candles={
            "15m": make_frame(70_000, 4, breakout=True),
            "4h": make_frame(62_000, 120),
        },
    )
    candidate, _ = MarketScanner(
        ScannerConfig(risk_per_trade_pct=5, max_risk_per_trade_pct=5, volatility_adjust_risk=False)
    ).scan([snapshot], RiskState(equity=1000)).tradable[0]
    first_plan = build_position_plan(candidate, RiskState(equity=1000), 5)
    second_plan = build_position_plan(candidate, RiskState(equity=1100), 5)
    assert first_plan.account_risk == 50
    assert second_plan.account_risk == 55


if __name__ == "__main__":
    test_scanner_finds_only_a_plus_breakout()
    test_risk_kill_switch_blocks_trading()
    test_liquidity_filter_rejects_low_quality_market()
    test_paper_engine_accounts_for_fees_and_tp()
    test_position_sizing_compounds_from_current_equity()
    print("self tests passed")
