from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks/trades"
DEFAULT_START = "2025-01-01T00:00:00Z"
DEFAULT_END = "2025-12-31T23:59:59Z"


def _chunk_symbols(symbols: Iterable[str], chunk_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for symbol in symbols:
        batch.append(symbol)
        if len(batch) == chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _alpaca_credentials() -> tuple[str, str]:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise EnvironmentError(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY before requesting Alpaca data."
        )
    return api_key, secret_key


def _alpaca_request(
        symbols: list[str],
        start: str,
        end: str,
        limit: int,
        page_token: str | None,
        feed: str,
        sort: str,
) -> dict:
    api_key, secret_key = _alpaca_credentials()
    params = {
        "symbols": ",".join(symbols),
        "start": start,
        "end": end,
        "limit": limit,
        "feed": feed,
        "sort": sort,
    }
    if page_token:
        params["page_token"] = page_token

    request = Request(
        f"{ALPACA_DATA_URL}?{urlencode(params)}",
        headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Accept": "application/json",
        },
    )
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _normalize_trade_data(payload: dict) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, trades in payload.get("trades", {}).items():
        if not trades:
            continue
        frame = pd.DataFrame(trades).rename(
            columns={
                "t": "timestamp",
                "x": "exchange",
                "p": "price",
                "s": "size",
                "c": "conditions",
                "i": "trade_id",
                "z": "tape",
            }
        )
        frame["symbol"] = symbol
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timestamp",
                "price",
                "size",
                "exchange",
                "conditions",
                "trade_id",
                "tape",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def fetch_historical_trades(
        symbols: str | Iterable[str],
        start: str = DEFAULT_START,
        end: str = DEFAULT_END,
        *,
        feed: str = "sip",
        limit: int = 10_000,
        sort: str = "asc",
        symbol_batch_size: int = 25,
        output_path: str | Path | None = None,
) -> pd.DataFrame:
    if isinstance(symbols, str):
        normalized_symbols = [symbols]
    else:
        normalized_symbols = sorted({symbol.upper() for symbol in symbols})
    if not normalized_symbols:
        raise ValueError("At least one symbol is required.")

    all_frames: list[pd.DataFrame] = []
    for batch in _chunk_symbols(normalized_symbols, symbol_batch_size):
        page_token: str | None = None
        while True:
            payload = _alpaca_request(
                symbols=batch,
                start=start,
                end=end,
                limit=limit,
                page_token=page_token,
                feed=feed,
                sort=sort,
            )
            all_frames.append(_normalize_trade_data(payload))
            page_token = payload.get("next_page_token")
            if not page_token:
                break

    if all_frames:
        trades = pd.concat(all_frames, ignore_index=True)
        trades = trades.sort_values(
            ["symbol", "timestamp", "trade_id"], kind="stable"
        ).reset_index(drop=True)
    else:
        trades = _normalize_trade_data({})

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_parquet(output_path, index=False)

    return trades


def load_historical_trades(output_path: str | Path) -> pd.DataFrame:
    output_path = Path(output_path)
    if not output_path.exists():
        raise FileNotFoundError(f"Historical trades file not found: {output_path}")
    return pd.read_parquet(output_path)
