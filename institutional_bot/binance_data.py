from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import TimeframeFrame


@dataclass(frozen=True)
class Candle:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float


@dataclass(frozen=True)
class BinanceMarket:
    symbol: str
    quote_volume: float
    spread_bps: float
    funding_rate: float


@dataclass(frozen=True)
class OpenInterestPoint:
    timestamp: datetime
    open_interest: float


class BinanceFuturesClient:
    def __init__(self, base_url: str = "https://fapi.binance.com", timeout: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def top_usdt_perp_markets(self, limit: int = 140, min_quote_volume: float = 50_000_000) -> list[BinanceMarket]:
        tickers = self._get("/fapi/v1/ticker/24hr")
        exchange_info = self._get("/fapi/v1/exchangeInfo")
        eligible = {
            item["symbol"]: item
            for item in exchange_info.get("symbols", [])
            if item.get("status") == "TRADING"
            and item.get("contractType") == "PERPETUAL"
            and item.get("quoteAsset") == "USDT"
            and item.get("marginAsset") == "USDT"
        }
        books = {item["symbol"]: item for item in self._get("/fapi/v1/ticker/bookTicker")}
        premiums = {item["symbol"]: item for item in self._get("/fapi/v1/premiumIndex")}
        markets: list[BinanceMarket] = []
        for ticker in tickers:
            symbol = ticker.get("symbol", "")
            info = eligible.get(symbol)
            if info is None or not self._is_supported_crypto_usdt_perp(symbol, info):
                continue
            quote_volume = float(ticker.get("quoteVolume") or 0)
            if quote_volume < min_quote_volume:
                continue
            book = books.get(symbol, {})
            bid = float(book.get("bidPrice") or 0)
            ask = float(book.get("askPrice") or 0)
            mid = (bid + ask) / 2 if bid and ask else float(ticker.get("lastPrice") or 0)
            spread_bps = ((ask - bid) / mid * 10_000) if mid and bid and ask else 10.0
            funding_rate = float(premiums.get(symbol, {}).get("lastFundingRate") or 0)
            markets.append(
                BinanceMarket(
                    symbol=symbol,
                    quote_volume=quote_volume,
                    spread_bps=spread_bps,
                    funding_rate=funding_rate,
                )
            )
        markets.sort(key=lambda market: market.quote_volume, reverse=True)
        return markets[:limit]

    def klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        limit: int = 1500,
    ) -> list[Candle]:
        candles: list[Candle] = []
        cursor = start
        step = _interval_delta_ms(interval)
        while cursor < end:
            payload = self._get(
                "/fapi/v1/klines",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": _ms(cursor),
                    "endTime": _ms(end),
                    "limit": limit,
                },
            )
            if not payload:
                break
            batch = [
                Candle(
                    open_time=datetime.fromtimestamp(item[0] / 1000, tz=timezone.utc),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    quote_volume=float(item[7]),
                )
                for item in payload
            ]
            candles.extend(batch)
            if len(payload) < limit:
                break
            cursor = datetime.fromtimestamp((payload[-1][0] + step) / 1000, tz=timezone.utc)
        return candles

    def open_interest_change_pct(self, symbol: str, start: datetime, end: datetime, period: str = "15m") -> float:
        try:
            payload = self._get(
                "/futures/data/openInterestHist",
                {
                    "symbol": symbol,
                    "period": period,
                    "startTime": _ms(start),
                    "endTime": _ms(end),
                    "limit": 30,
                },
            )
        except Exception:
            return 0.0
        if len(payload) < 2:
            return 0.0
        first = float(payload[0].get("sumOpenInterest") or 0)
        last = float(payload[-1].get("sumOpenInterest") or 0)
        if first <= 0:
            return 0.0
        return (last - first) / first * 100

    def open_interest_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        period: str = "1h",
        limit: int = 500,
    ) -> list[OpenInterestPoint]:
        points: list[OpenInterestPoint] = []
        cursor = start
        step = _period_delta_ms(period)
        while cursor < end:
            payload = self._get(
                "/futures/data/openInterestHist",
                {
                    "symbol": symbol,
                    "period": period,
                    "startTime": _ms(cursor),
                    "endTime": _ms(end),
                    "limit": limit,
                },
            )
            if not payload:
                break
            batch = [
                OpenInterestPoint(
                    timestamp=datetime.fromtimestamp(item["timestamp"] / 1000, tz=timezone.utc),
                    open_interest=float(item.get("sumOpenInterest") or 0),
                )
                for item in payload
            ]
            points.extend(batch)
            if len(payload) < limit:
                break
            cursor = datetime.fromtimestamp((payload[-1]["timestamp"] + step) / 1000, tz=timezone.utc)
        return points

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "institutional-paper-bot/0.1"})
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _is_supported_crypto_usdt_perp(symbol: str, info: dict[str, Any]) -> bool:
        excluded_fragments = (
            "1000",
            "MEME",
            "PEPE",
            "BONK",
            "FLOKI",
            "DOGE",
            "SHIB",
            "TURBO",
            "WIF",
            "PUMP",
            "XAG",
            "XAU",
            "TSLA",
            "AAPL",
            "NVDA",
            "MSTR",
            "COIN",
        )
        blocked_subtypes = {"COMMODITY", "METAL", "STOCK", "EQUITY", "FOREX"}
        subtypes = {str(value).upper() for value in info.get("underlyingSubType", [])}
        underlying_type = str(info.get("underlyingType", "")).upper()
        return (
            symbol.endswith("USDT")
            and not any(fragment in symbol for fragment in excluded_fragments)
            and underlying_type not in blocked_subtypes
            and not (subtypes & blocked_subtypes)
        )


def frame_from_candles(candles: list[Candle]) -> TimeframeFrame:
    return TimeframeFrame(
        open=tuple(candle.open for candle in candles),
        high=tuple(candle.high for candle in candles),
        low=tuple(candle.low for candle in candles),
        close=tuple(candle.close for candle in candles),
        volume=tuple(candle.volume for candle in candles),
    )


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _interval_delta_ms(interval: str) -> int:
    minutes = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "12h": 720,
        "1d": 1440,
    }[interval]
    return minutes * 60 * 1000


def _period_delta_ms(period: str) -> int:
    return _interval_delta_ms(period)
