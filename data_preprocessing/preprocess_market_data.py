from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarResult:
    sample: pd.DataFrame
    ohlcv: pd.DataFrame


def _ewma(values: list[float], span: int) -> float:
    if not values:
        return 0.0
    return float(pd.Series(values, dtype=float).ewm(span=span, adjust=False).mean().iloc[-1])


def _prepare_trade_data(
        trades: pd.DataFrame,
        *,
        timestamp_col: str = "timestamp",
        price_col: str = "price",
        volume_col: str = "size",
        symbol_col: str = "symbol",
) -> pd.DataFrame:
    df = trades.copy()
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        df = df.sort_values([timestamp_col], kind="stable")
        df = df.set_index(timestamp_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Trades must include a timestamp column or a DatetimeIndex.")

    df.index.name = "timestamp"
    df[price_col] = df[price_col].astype(float)
    df[volume_col] = df[volume_col].astype(float)
    if symbol_col not in df.columns:
        df[symbol_col] = "UNKNOWN"

    price_diff = df[price_col].diff()
    tick_sign = np.sign(price_diff).replace(0.0, np.nan).ffill().fillna(1.0)
    df["tick_sign"] = tick_sign.astype(float)
    df["dollar_value"] = df[price_col] * df[volume_col]
    df["signed_tick"] = df["tick_sign"]
    df["signed_volume"] = df["tick_sign"] * df[volume_col]
    df["signed_dollar_value"] = df["tick_sign"] * df["dollar_value"]
    return df


def _build_ohlcv_bars(
        trades: pd.DataFrame,
        bar_end_indices: list[int],
        *,
        price_col: str,
        volume_col: str,
) -> BarResult:
    if not bar_end_indices:
        empty_sample = trades.iloc[0:0].copy()
        empty_ohlcv = pd.DataFrame(
            columns=[
                "start",
                "end",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "dollar_value",
                "ticks",
                "buy_volume",
                "sell_volume",
            ]
        )
        return BarResult(sample=empty_sample, ohlcv=empty_ohlcv)

    sample = trades.iloc[bar_end_indices].copy()
    rows: list[dict] = []
    start_idx = 0
    for end_idx in bar_end_indices:
        window = trades.iloc[start_idx: end_idx + 1]
        rows.append(
            {
                "start": window.index[0],
                "end": window.index[-1],
                "symbol": window["symbol"].iloc[-1],
                "open": float(window[price_col].iloc[0]),
                "high": float(window[price_col].max()),
                "low": float(window[price_col].min()),
                "close": float(window[price_col].iloc[-1]),
                "volume": float(window[volume_col].sum()),
                "dollar_value": float(window["dollar_value"].sum()),
                "ticks": int(len(window)),
                "buy_volume": float(window.loc[window["tick_sign"] > 0, volume_col].sum()),
                "sell_volume": float(window.loc[window["tick_sign"] < 0, volume_col].sum()),
            }
        )
        start_idx = end_idx + 1

    ohlcv = pd.DataFrame(rows).set_index("end")
    return BarResult(sample=sample, ohlcv=ohlcv)


def _compute_threshold_bar_end_indices(values: pd.Series, threshold: float) -> list[int]:
    if threshold <= 0:
        raise ValueError("Threshold must be positive.")
    cumulative_value = 0.0
    indices: list[int] = []
    for idx, value in enumerate(values.astype(float).to_numpy()):
        cumulative_value += value
        if cumulative_value >= threshold:
            indices.append(idx)
            cumulative_value = 0.0
    return indices


def get_tick_bars(
        trades: pd.DataFrame,
        threshold: int,
        *,
        price_col: str = "price",
        volume_col: str = "size",
) -> BarResult:
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_threshold_bar_end_indices(pd.Series(1.0, index=prepared.index), threshold)
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_volume_bars(
        trades: pd.DataFrame,
        threshold: float,
        *,
        price_col: str = "price",
        volume_col: str = "size",
) -> BarResult:
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_threshold_bar_end_indices(prepared[volume_col], threshold)
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_dollar_bars(
        trades: pd.DataFrame,
        threshold: float,
        *,
        price_col: str = "price",
        volume_col: str = "size",
) -> BarResult:
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_threshold_bar_end_indices(prepared["dollar_value"], threshold)
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def _compute_imbalance_bar_end_indices(
        prepared: pd.DataFrame,
        imbalance_col: str,
        *,
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
        min_exp_num_ticks: int = 10,
        max_exp_num_ticks: int = 100_000,
) -> list[int]:
    values = prepared[imbalance_col].astype(float).to_numpy()
    if len(values) == 0:
        return []

    seed = values[: min(len(values), expected_num_ticks_init)]
    non_zero_seed = seed[np.abs(seed) > 0]
    expected_imbalance = float(non_zero_seed.mean()) if len(non_zero_seed) else 1.0
    expected_num_ticks = float(max(min_exp_num_ticks, min(expected_num_ticks_init, len(values))))

    indices: list[int] = []
    bar_sizes: list[int] = []
    mean_bar_imbalances: list[float] = []

    cumulative_imbalance = 0.0
    ticks_in_bar = 0
    current_values: list[float] = []

    for idx, value in enumerate(values):
        cumulative_imbalance += value
        ticks_in_bar += 1
        current_values.append(value)
        threshold = max(1e-12, expected_num_ticks * abs(expected_imbalance))

        if abs(cumulative_imbalance) >= threshold:
            indices.append(idx)
            bar_sizes.append(ticks_in_bar)
            mean_bar_imbalances.append(float(np.mean(current_values)))

            expected_num_ticks = float(
                np.clip(
                    _ewma(bar_sizes[-expected_window:], expected_window),
                    min_exp_num_ticks,
                    max_exp_num_ticks,
                )
            )
            expected_imbalance = _ewma(mean_bar_imbalances[-expected_window:], expected_window)
            if abs(expected_imbalance) < 1e-12:
                expected_imbalance = 1.0

            cumulative_imbalance = 0.0
            ticks_in_bar = 0
            current_values = []

    return indices


def get_tick_imbalance_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
) -> BarResult:
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_imbalance_bar_end_indices(
        prepared,
        "signed_tick",
        expected_num_ticks_init=expected_num_ticks_init,
        expected_window=expected_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_volume_imbalance_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
) -> BarResult:
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_imbalance_bar_end_indices(
        prepared,
        "signed_volume",
        expected_num_ticks_init=expected_num_ticks_init,
        expected_window=expected_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_dollar_imbalance_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
) -> BarResult:
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_imbalance_bar_end_indices(
        prepared,
        "signed_dollar_value",
        expected_num_ticks_init=expected_num_ticks_init,
        expected_window=expected_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def _compute_run_bar_end_indices(
        prepared: pd.DataFrame,
        imbalance_col: str,
        *,
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
        min_exp_num_ticks: int = 10,
        max_exp_num_ticks: int = 100_000,
) -> list[int]:
    values = prepared[imbalance_col].astype(float).to_numpy()
    if len(values) == 0:
        return []

    initial = values[: min(len(values), expected_num_ticks_init)]
    initial_buy = initial[initial > 0]
    initial_sell = -initial[initial < 0]
    expected_buy = float(initial_buy.mean()) if len(initial_buy) else 1.0
    expected_sell = float(initial_sell.mean()) if len(initial_sell) else 1.0
    expected_buy_prob = float((initial > 0).mean()) if len(initial) else 0.5
    expected_num_ticks = float(max(min_exp_num_ticks, min(expected_num_ticks_init, len(values))))

    indices: list[int] = []
    bar_sizes: list[int] = []
    bar_buy_probs: list[float] = []
    bar_buy_means: list[float] = []
    bar_sell_means: list[float] = []

    cumulative_buy = 0.0
    cumulative_sell = 0.0
    ticks_in_bar = 0
    buy_values: list[float] = []
    sell_values: list[float] = []

    for idx, value in enumerate(values):
        ticks_in_bar += 1
        if value > 0:
            cumulative_buy += value
            buy_values.append(value)
        elif value < 0:
            cumulative_sell += -value
            sell_values.append(-value)

        threshold = max(
            1e-12,
            expected_num_ticks
            * max(expected_buy_prob * expected_buy, (1.0 - expected_buy_prob) * expected_sell),
        )

        if max(cumulative_buy, cumulative_sell) >= threshold:
            indices.append(idx)
            bar_sizes.append(ticks_in_bar)
            bar_buy_probs.append(len(buy_values) / ticks_in_bar if ticks_in_bar else 0.5)
            bar_buy_means.append(float(np.mean(buy_values)) if buy_values else 0.0)
            bar_sell_means.append(float(np.mean(sell_values)) if sell_values else 0.0)

            expected_num_ticks = float(
                np.clip(
                    _ewma(bar_sizes[-expected_window:], expected_window),
                    min_exp_num_ticks,
                    max_exp_num_ticks,
                )
            )
            expected_buy_prob = _ewma(bar_buy_probs[-expected_window:], expected_window)
            expected_buy = max(_ewma(bar_buy_means[-expected_window:], expected_window), 1e-12)
            expected_sell = max(_ewma(bar_sell_means[-expected_window:], expected_window), 1e-12)

            cumulative_buy = 0.0
            cumulative_sell = 0.0
            ticks_in_bar = 0
            buy_values = []
            sell_values = []

    return indices


def get_tick_run_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
) -> BarResult:
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_run_bar_end_indices(
        prepared,
        "signed_tick",
        expected_num_ticks_init=expected_num_ticks_init,
        expected_window=expected_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_volume_run_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
) -> BarResult:
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_run_bar_end_indices(
        prepared,
        "signed_volume",
        expected_num_ticks_init=expected_num_ticks_init,
        expected_window=expected_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_dollar_run_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
) -> BarResult:
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_run_bar_end_indices(
        prepared,
        "signed_dollar_value",
        expected_num_ticks_init=expected_num_ticks_init,
        expected_window=expected_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)
