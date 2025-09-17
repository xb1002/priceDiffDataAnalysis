# -*- coding: utf-8 -*-
"""Signal duration analysis for the basic moving-average spread strategy.

Per AGENTS.md task 5:
- Load merged 1s BBO data for Binance (main) and Bitget (hedge).
- Compute 4h rolling mean/std ratios as in backtest.py.
- Track continuous stretches of open/close signals for long/short sides.
- Output per-symbol statistics and a summary CSV across all symbols.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest import (
    add_spread_stats,
    load_merged_orderbook,
)


@dataclass
class AnalysisParams:
    window: str = "4h"
    min_periods: int = 600
    profit_threshold: float = 0.0004
    fee_main: float = 0.0
    fee_hedge: float = 0.00015


def discover_symbols(binance_dir: str, bitget_dir: str, symbols: Optional[Iterable[str]]) -> List[str]:
    if symbols:
        return list(symbols)
    if not (os.path.isdir(binance_dir) and os.path.isdir(bitget_dir)):
        return []
    bfiles = {f[:-4] for f in os.listdir(binance_dir) if f.endswith(".csv")}
    gfiles = {f[:-4] for f in os.listdir(bitget_dir) if f.endswith(".csv")}
    return sorted(bfiles.intersection(gfiles))


def _load_raw_orderbook(path: str, bid_col: str, ask_col: str, ts_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[bid_col, ask_col, ts_col])
    df.columns = ["bid", "ask", "timestamp"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[~df["timestamp"].duplicated(keep="first")]
    df = df[df["timestamp"].notna()].sort_values("timestamp")
    return df.set_index("timestamp")


def compute_effective_coverage(
    symbol: str,
    binance_dir: str,
    bitget_dir: str,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, int, int, float]]:
    binance_fp = os.path.join(binance_dir, f"{symbol}.csv")
    bitget_fp = os.path.join(bitget_dir, f"{symbol}.csv")
    if not (os.path.exists(binance_fp) and os.path.exists(bitget_fp)):
        return None

    try:
        main = _load_raw_orderbook(binance_fp, "b", "a", "T")
        hedge = _load_raw_orderbook(bitget_fp, "bidPr", "askPr", "ts")
    except Exception as exc:
        print(f"{symbol}: 原始数据读取失败 ({exc})")
        return None

    if main.empty or hedge.empty:
        return None

    start = max(main.index[0], hedge.index[0]).floor("1s")
    end = min(main.index[-1], hedge.index[-1]).floor("1s")
    if end <= start:
        return None

    time_index = pd.date_range(start=start, end=end, freq="1s")
    expected_seconds = len(time_index)

    main_1s = main.resample("1s").last().reindex(time_index)
    hedge_1s = hedge.resample("1s").last().reindex(time_index)

    mask = (
        main_1s[["bid", "ask"]].notna().all(axis=1)
        & hedge_1s[["bid", "ask"]].notna().all(axis=1)
    )
    available_seconds = int(mask.sum())
    coverage = available_seconds / expected_seconds if expected_seconds > 0 else 0.0
    return start, end, expected_seconds, available_seconds, coverage


def compute_signal_series(df: pd.DataFrame, params: AnalysisParams) -> Dict[str, pd.Series]:
    fee_buffer = 2.0 * params.fee_hedge
    valid_stats = (
        np.isfinite(df["bid_mean"]) &
        np.isfinite(df["ask_mean"]) &
        np.isfinite(df["bid_std"]) &
        np.isfinite(df["ask_std"])
    )
    long_margin = (df["ask_mean"] + df["ask_std"]) - df["bid_bid"] - fee_buffer
    short_margin = df["ask_ask"] - (df["bid_mean"] - df["bid_std"]) - fee_buffer

    long_open = valid_stats & np.isfinite(df["bid_bid"]) & (long_margin > params.profit_threshold)
    short_open = valid_stats & np.isfinite(df["ask_ask"]) & (short_margin > params.profit_threshold)

    long_close = valid_stats & np.isfinite(df["ask_ask"]) & (df["ask_ask"] >= (df["ask_mean"] + df["ask_std"]))
    short_close = valid_stats & np.isfinite(df["bid_bid"]) & (df["bid_bid"] <= (df["bid_mean"] - df["bid_std"]))

    signals = {
        "long_open": long_open.fillna(False),
        "short_open": short_open.fillna(False),
        "long_close": long_close.fillna(False),
        "short_close": short_close.fillna(False),
        "open_any": (long_open | short_open).fillna(False),
        "close_any": (long_close | short_close).fillna(False),
    }
    return {k: v.astype(bool) for k, v in signals.items()}


def compute_signal_stats(signal_series: pd.Series) -> Dict[str, float]:
    if signal_series.empty:
        return {
            "count": 0,
            "total_seconds": 0.0,
            "avg_seconds": 0.0,
            "max_seconds": 0.0,
            "min_seconds": 0.0,
        }

    sr = signal_series.fillna(False).astype(bool)
    groups = sr.ne(sr.shift()).cumsum()
    durations = [int(len(group)) for _, group in sr.groupby(groups) if group.iloc[0]]

    if not durations:
        return {
            "count": 0,
            "total_seconds": 0.0,
            "avg_seconds": 0.0,
            "max_seconds": 0.0,
            "min_seconds": 0.0,
        }

    total = float(sum(durations))
    return {
        "count": int(len(durations)),
        "total_seconds": total,
        "avg_seconds": total / len(durations),
        "max_seconds": float(max(durations)),
        "min_seconds": float(min(durations)),
    }


def analyze_symbol(
    symbol: str,
    params: AnalysisParams,
    binance_dir: str,
    bitget_dir: str,
) -> Optional[Dict[str, object]]:
    coverage_info = compute_effective_coverage(symbol, binance_dir, bitget_dir)
    if coverage_info is None:
        print(f"{symbol}: 无法计算数据覆盖率，跳过")
        return None
    start_ts, end_ts, expected_seconds, available_seconds, coverage = coverage_info
    if expected_seconds <= 0:
        print(f"{symbol}: 时间跨度异常，跳过")
        return None
    if coverage < 0.7:
        print(f"{symbol}: 数据覆盖率 {coverage:.2%} < 70%，跳过")
        return None

    try:
        df = load_merged_orderbook(symbol, binance_dir, bitget_dir, resample="1s")
    except FileNotFoundError as exc:
        print(f"{symbol}: data missing ({exc})")
        return None
    except Exception as exc:
        print(f"{symbol}: failed to load data ({exc})")
        return None

    if df.empty:
        print(f"{symbol}: merged dataframe is empty")
        return None

    df = df.sort_index().loc[start_ts:end_ts]
    df = add_spread_stats(df, window=params.window, min_periods=params.min_periods)
    df = df.sort_index()
    signals = compute_signal_series(df, params)

    time_span_seconds = float(expected_seconds)

    metrics: Dict[str, object] = {
        "symbol": symbol,
        "observations": int(len(df)),
        "raw_expected_seconds": expected_seconds,
        "raw_available_seconds": available_seconds,
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "coverage": coverage,
        "time_span_seconds": time_span_seconds,
    }

    for name, series in signals.items():
        stats = compute_signal_stats(series)
        metrics[f"{name}_count"] = stats["count"]
        metrics[f"{name}_total_seconds"] = stats["total_seconds"]
        metrics[f"{name}_avg_seconds"] = stats["avg_seconds"]
        metrics[f"{name}_max_seconds"] = stats["max_seconds"]
        metrics[f"{name}_min_seconds"] = stats["min_seconds"]

    metrics["long_open_ratio"] = (
        metrics["long_open_total_seconds"] / time_span_seconds
        if time_span_seconds > 0
        else 0.0
    )
    metrics["short_open_ratio"] = (
        metrics["short_open_total_seconds"] / time_span_seconds
        if time_span_seconds > 0
        else 0.0
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="统计均线回归策略信号持续时间")
    parser.add_argument("--symbols", nargs="*", help="指定要分析的交易对，留空则分析共有的数据集")
    parser.add_argument("--binance-dir", default="data/binance", help="Binance 原始数据目录")
    parser.add_argument("--bitget-dir", default="data/bitget", help="Bitget 原始数据目录")
    parser.add_argument("--window", default="4h", help="滚动均线窗口，默认4h")
    parser.add_argument("--min-periods", type=int, default=600, help="滚动统计最小样本量")
    parser.add_argument("--profit-threshold", type=float, default=0.0004, help="开仓利润阈值")
    parser.add_argument("--fee-main", type=float, default=0.0, help="主交易所手续费")
    parser.add_argument("--fee-hedge", type=float, default=0.00015, help="对冲交易所手续费")
    parser.add_argument("--output", default="data/backtest/compare/signal_time_summary.csv", help="汇总结果输出路径")

    args = parser.parse_args()

    sparams = AnalysisParams(
        window=args.window,
        min_periods=args.min_periods,
        profit_threshold=args.profit_threshold,
        fee_main=args.fee_main,
        fee_hedge=args.fee_hedge,
    )

    symbols = discover_symbols(args.binance_dir, args.bitget_dir, args.symbols)
    if not symbols:
        print("未找到可用的交易对数据")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results: List[Dict[str, object]] = []
    for sym in symbols:
        metrics = analyze_symbol(sym, sparams, args.binance_dir, args.bitget_dir)
        if not metrics:
            continue
        results.append(metrics)
        print(
            f"{sym}: span {metrics['time_span_seconds']:.0f}s | raw {metrics['raw_available_seconds']}/{metrics['raw_expected_seconds']}s "
            f"({metrics['coverage']:.2%}) | "
            f"long_open {metrics['long_open_count']} signals (~{metrics['long_open_ratio']:.2%}); "
            f"short_open {metrics['short_open_count']} signals (~{metrics['short_open_ratio']:.2%})"
        )

    if not results:
        print("所有交易对均无法计算信号统计")
        return

    summary_df = pd.DataFrame(results)
    summary_df.sort_values("symbol", inplace=True)
    summary_df.to_csv(args.output, index=False)
    print(f"信号统计已写入 {args.output}")


if __name__ == "__main__":
    main()
