# -*- coding: utf-8 -*-
"""
Task 4: 全币种回测 (portfolio across all pairs)

Implements a multi-pair spread-arbitrage backtest using the same core
logic as backtest.py, extended to:
- Preprocess all symbols to 1s BBO and rolling stats and save under data/bbo/
- Simulate a portfolio with initial capital and per-pair max weight
- Each second, compute expected margins for all pairs and open positions
  in the top-N by expected return opportunity subject to capacity
- Close positions using either reverse/line/both as in backtest.py
- Output overall portfolio metrics and equity curve

Notes:
- Data files expected like backtest.py: data/binance/<sym>.csv with columns
  b,a,T (ms); data/bitget/<sym>.csv with bidPr,askPr,ts (ms)
- The preprocessed files are written to data/bbo/<sym>_bbo_1s.csv with columns:
  bid_binance, ask_binance, bid_bitget, ask_bitget, bid_bid, ask_ask,
  bid_mean, bid_std, ask_mean, ask_std
"""

from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd


# -----------------------------
# Shared data utilities (aligned with backtest.py)
# -----------------------------

def load_merged_orderbook(
    symbol: str,
    binance_dir: str = "data/binance",
    bitget_dir: str = "data/bitget",
    resample: str = "1s",
) -> pd.DataFrame:
    binance_fp = os.path.join(binance_dir, f"{symbol}.csv")
    bitget_fp = os.path.join(bitget_dir, f"{symbol}.csv")
    if not (os.path.exists(binance_fp) and os.path.exists(bitget_fp)):
        raise FileNotFoundError(f"Missing CSV for {symbol}: {binance_fp} or {bitget_fp}")

    bdf = pd.read_csv(binance_fp)
    gdf = pd.read_csv(bitget_fp)

    bdf = bdf[["b", "a", "T"]].copy()
    bdf.columns = ["bid", "ask", "timestamp"]
    bdf["timestamp"] = pd.to_datetime(bdf["timestamp"], unit="ms")
    bdf.set_index("timestamp", inplace=True)

    gdf = gdf[["bidPr", "askPr", "ts"]].copy()
    gdf.columns = ["bid", "ask", "timestamp"]
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], unit="ms")
    gdf.set_index("timestamp", inplace=True)

    bdf = bdf[~bdf.index.duplicated(keep="first")].sort_index()
    gdf = gdf[~gdf.index.duplicated(keep="first")].sort_index()
    bdf = bdf[bdf.index.notna()].dropna()
    gdf = gdf[gdf.index.notna()].dropna()

    b1 = bdf.resample(resample).ffill().dropna()
    g1 = gdf.resample(resample).ffill().dropna()

    merged = b1.merge(g1, left_index=True, right_index=True, suffixes=("_binance", "_bitget"))
    merged = merged[(merged[["bid_binance", "ask_binance", "bid_bitget", "ask_bitget"]] > 0).all(axis=1)]
    return merged


def add_spread_stats(
    df: pd.DataFrame,
    window: str = "4h",
    min_periods: int = 600,
) -> pd.DataFrame:
    out = df.copy()
    out["bid_bid"] = out["bid_binance"] / out["bid_bitget"]
    out["ask_ask"] = out["ask_binance"] / out["ask_bitget"]

    roll_bid = out["bid_bid"].rolling(window=window, min_periods=min_periods)
    roll_ask = out["ask_ask"].rolling(window=window, min_periods=min_periods)
    out["bid_mean"] = roll_bid.mean()
    out["bid_std"] = roll_bid.std(ddof=0)
    out["ask_mean"] = roll_ask.mean()
    out["ask_std"] = roll_ask.std(ddof=0)
    return out


# -----------------------------
# Strategy primitives (aligned with backtest.py)
# -----------------------------

CloseMode = Literal["reverse", "line", "both"]


@dataclass
class Params:
    profit_threshold: float = 0.0004
    fee_main: float = 0.0
    fee_hedge: float = 0.00015
    close_mode: CloseMode = "both"
    window: str = "4h"
    min_periods: int = 600


@dataclass
class Position:
    symbol: str
    side: Literal["long", "short"]
    entry_time: pd.Timestamp
    entry_main_price: float
    entry_hedge_price: float
    weight: float  # fraction of equity allocated to this position at entry


def estimate_entry_profit_margin(row: pd.Series, params: Params, side: Literal["long", "short"]) -> float:
    fee_buffer = 2.0 * params.fee_hedge
    if side == "long":
        margin = (row["ask_mean"] + row["ask_std"]) - row["bid_bid"]
    else:
        margin = row["ask_ask"] - (row["bid_mean"] - row["bid_std"])
    return float(margin - fee_buffer)


def should_open(row: pd.Series, params: Params, side: Literal["long", "short"]) -> bool:
    if not np.isfinite(row.get("bid_mean", np.nan)) or not np.isfinite(row.get("ask_mean", np.nan)):
        return False
    margin = estimate_entry_profit_margin(row, params, side)
    return margin > params.profit_threshold


def check_close_signals(row: pd.Series, params: Params, side: Literal["long", "short"]) -> Tuple[bool, bool]:
    reverse_signal = should_open(row, params, "short" if side == "long" else "long")
    if side == "long":
        line_signal = (row["ask_ask"] >= (row["ask_mean"] + row["ask_std"])) if np.isfinite(row.get("ask_mean", np.nan)) else False
    else:
        line_signal = (row["bid_bid"] <= (row["bid_mean"] - row["bid_std"])) if np.isfinite(row.get("bid_mean", np.nan)) else False
    return reverse_signal, line_signal


def execute_pnl(
    pos: Position,
    exit_time: pd.Timestamp,
    exit_main_price: float,
    exit_hedge_price: float,
    params: Params,
) -> Tuple[float, float]:
    fh = params.fee_hedge
    if pos.side == "long":
        Pm, Ph = pos.entry_main_price, pos.entry_hedge_price
        Qm, Qh = exit_main_price, exit_hedge_price
        pnl = (Ph * (1 - fh) - Pm) + (Qm - Qh * (1 + fh))
        ret = pnl / max(Pm, 1e-12)
    else:
        Pm, Ph = pos.entry_main_price, pos.entry_hedge_price
        Qm, Qh = exit_main_price, exit_hedge_price
        pnl = (Pm - Ph * (1 + fh)) + (Qh * (1 - fh) - Qm)
        ret = pnl / max(Pm, 1e-12)
    return pnl, ret


# -----------------------------
# Preprocessing: build 1s BBO and stats for all pairs
# -----------------------------

def preprocess_pairs(
    symbols: List[str],
    binance_dir: str,
    bitget_dir: str,
    bbo_dir: str,
    window: str,
    min_periods: int,
) -> List[str]:
    os.makedirs(bbo_dir, exist_ok=True)
    ok_syms: List[str] = []
    for sym in symbols:
        try:
            df = load_merged_orderbook(sym, binance_dir, bitget_dir, resample="1s")
            df = add_spread_stats(df, window=window, min_periods=min_periods)
            # Persist only needed columns to save space
            cols = [
                "bid_binance", "ask_binance", "bid_bitget", "ask_bitget",
                "bid_bid", "ask_ask", "bid_mean", "bid_std", "ask_mean", "ask_std"
            ]
            out = df[cols].copy()
            out_path = os.path.join(bbo_dir, f"{sym}_bbo_1s.csv")
            out.to_csv(out_path)
            ok_syms.append(sym)
            # Release memory explicitly
            del df
            del out
        except Exception as e:
            print(f"Preprocess skip {sym}: {e}")
    return ok_syms


def load_bbo(sym: str, bbo_dir: str) -> pd.DataFrame:
    fp = os.path.join(bbo_dir, f"{sym}_bbo_1s.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    # Ensure monotonic index and sanity
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


# -----------------------------
# Portfolio backtest
# -----------------------------

@dataclass
class PortfolioParams:
    initial_capital: float = 10000.0
    max_weight_per_pair: float = 0.2
    top_n: int = 3


def compute_portfolio_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
    metrics = {
        "trades": int(len(trades_df)),
        "wins": int((trades_df["weighted_ret"] > 0).sum()) if not trades_df.empty else 0,
        "win_rate": float(((trades_df["weighted_ret"] > 0).mean()) if not trades_df.empty else 0.0),
        "total_return": float((equity_df["equity"].iloc[-1] - equity_df["equity"].iloc[0]) if not equity_df.empty else 0.0),
        "total_ret_pct": float((equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1.0) if not equity_df.empty else 0.0),
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "duration_days": 0.0,
        "avg_holding_minutes": float(trades_df["holding_minutes"].mean()) if (not trades_df.empty and "holding_minutes" in trades_df.columns) else 0.0,
    }
    if equity_df.empty:
        return metrics

    eq = equity_df["equity"].astype(float)
    start, end = eq.index[0], eq.index[-1]
    days = max((end - start).total_seconds() / (3600 * 24), 1e-12)
    metrics["duration_days"] = days

    roll_max = eq.cummax()
    drawdown = (eq - roll_max) / roll_max
    metrics["max_drawdown"] = float(drawdown.min())

    daily = eq.resample("1D").last().pct_change().dropna()
    if len(daily) > 1 and daily.std(ddof=0) > 0:
        sharpe = float(daily.mean() / daily.std(ddof=0) * math.sqrt(365.0))
        metrics["sharpe"] = sharpe
    else:
        metrics["sharpe"] = 0.0
    return metrics


def run_portfolio(
    symbols: List[str],
    bbo_dir: str,
    pparams: PortfolioParams,
    sparams: Params,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load all bbo dataframes
    series: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = load_bbo(sym, bbo_dir)
            series[sym] = df
        except Exception as e:
            print(f"Load skip {sym}: {e}")

    if not series:
        return pd.DataFrame(), pd.DataFrame()

    # Compute common time intersection to keep per-second updates manageable
    starts = [df.index[0] for df in series.values()]
    ends = [df.index[-1] for df in series.values()]
    start_time = max(starts)
    end_time = min(ends)
    if end_time <= start_time:
        print("No overlapping time range across symbols.")
        return pd.DataFrame(), pd.DataFrame()

    time_index = pd.date_range(start=start_time, end=end_time, freq="1s")

    equity = float(pparams.initial_capital)
    equity_times: List[pd.Timestamp] = []
    equity_vals: List[float] = []

    open_positions: Dict[str, Position] = {}
    used_weight: float = 0.0
    trades: List[dict] = []

    for ts in time_index:
        equity_times.append(ts)
        equity_vals.append(equity)

        # 1) Handle closes for existing positions
        to_close: List[str] = []
        for sym, pos in open_positions.items():
            row = series[sym].loc[ts]
            reverse_sig, line_sig = check_close_signals(row, sparams, pos.side)
            do_close = (
                (sparams.close_mode == "reverse" and reverse_sig) or
                (sparams.close_mode == "line" and line_sig) or
                (sparams.close_mode == "both" and (reverse_sig or line_sig))
            )
            if do_close:
                if pos.side == "long":
                    exit_main = float(row["ask_binance"])  # sell main at ask
                    exit_hedge = float(row["ask_bitget"])  # buy hedge at ask
                else:
                    exit_main = float(row["bid_binance"])  # buy main at bid
                    exit_hedge = float(row["bid_bitget"])  # sell hedge at bid
                pnl, ret = execute_pnl(pos, ts, exit_main, exit_hedge, sparams)
                equity *= (1.0 + pos.weight * ret)
                trades.append({
                    "symbol": sym,
                    "side": pos.side,
                    "entry_time": pos.entry_time,
                    "exit_time": ts,
                    "entry_main": pos.entry_main_price,
                    "entry_hedge": pos.entry_hedge_price,
                    "exit_main": exit_main,
                    "exit_hedge": exit_hedge,
                    "pnl": pnl,
                    "ret": ret,
                    "weight": pos.weight,
                    "weighted_ret": pos.weight * ret,
                    "holding_minutes": float((ts - pos.entry_time).total_seconds() / 60.0),
                })
                used_weight -= pos.weight
                used_weight = max(used_weight, 0.0)
                to_close.append(sym)
        for sym in to_close:
            del open_positions[sym]

        # 2) Compute open candidates if capacity available
        capacity = max(1.0 - used_weight, 0.0)
        if capacity > 1e-9:
            candidates: List[Tuple[str, float, str]] = []  # (symbol, margin, side)
            for sym, df in series.items():
                if sym in open_positions:
                    continue
                row = df.loc[ts]
                can_long = should_open(row, sparams, "long")
                can_short = should_open(row, sparams, "short")
                if not (can_long or can_short):
                    continue
                if can_long and not can_short:
                    side = "long"; margin = estimate_entry_profit_margin(row, sparams, "long")
                elif can_short and not can_long:
                    side = "short"; margin = estimate_entry_profit_margin(row, sparams, "short")
                else:
                    m_long = estimate_entry_profit_margin(row, sparams, "long")
                    m_short = estimate_entry_profit_margin(row, sparams, "short")
                    if m_long >= m_short:
                        side, margin = "long", m_long
                    else:
                        side, margin = "short", m_short
                candidates.append((sym, float(margin), side))

            if candidates:
                # sort by margin desc and take top-N
                candidates.sort(key=lambda x: x[1], reverse=True)
                take = candidates[: max(1, int(pparams.top_n))]
                for sym, _m, side in take:
                    if capacity <= 1e-9:
                        break
                    row = series[sym].loc[ts]
                    if side == "long":
                        entry_main = float(row["bid_binance"])  # buy main at bid
                        entry_hedge = float(row["bid_bitget"])  # sell hedge at bid
                    else:
                        entry_main = float(row["ask_binance"])  # sell main at ask
                        entry_hedge = float(row["ask_bitget"])  # buy hedge at ask

                    w = float(min(pparams.max_weight_per_pair, capacity))
                    pos = Position(
                        symbol=sym,
                        side=side,
                        entry_time=ts,
                        entry_main_price=entry_main,
                        entry_hedge_price=entry_hedge,
                        weight=w,
                    )
                    open_positions[sym] = pos
                    used_weight += w
                    capacity -= w

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame({"equity": equity_vals}, index=pd.to_datetime(equity_times))
    return trades_df, equity_df


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="全币种回测：多品种投资组合的价差套利回测")
    parser.add_argument("--binance-dir", default="data/binance", help="Binance 数据目录")
    parser.add_argument("--bitget-dir", default="data/bitget", help="Bitget 数据目录")
    parser.add_argument("--bbo-dir", default="data/bbo", help="预处理后的BBO输出目录")
    parser.add_argument("--output-dir", default="./data/backtest", help="输出目录")
    parser.add_argument("--max-symbols", type=int, default=50, help="最多读取的交易对数量（上限100）")

    # Trading/stat params
    parser.add_argument("--window", default="4h", help="回归线滚动窗口，默认4h")
    parser.add_argument("--min-periods", type=int, default=600, help="滚动统计最小样本数")
    parser.add_argument("--profit-threshold", type=float, default=0.0003, help="开仓利润阈值（比率单位）")
    parser.add_argument("--fee-main", type=float, default=0.0, help="主交易所手续费")
    parser.add_argument("--fee-hedge", type=float, default=0.00015, help="对冲交易所手续费")
    parser.add_argument("--close-mode", choices=["reverse", "line", "both"], default="both", help="平仓方式")

    # Portfolio params
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="初始资金（USDT）")
    parser.add_argument("--max-weight-per-pair", type=float, default=0.2, help="单品种最大权重（0-1）")
    parser.add_argument("--top-n", type=int, default=3, help="每秒选择前N个进行交易")

    args = parser.parse_args()

    # Discover symbols (intersection of dirs)
    if os.path.isdir(args.binance_dir) and os.path.isdir(args.bitget_dir):
        bfiles = {f[:-4] for f in os.listdir(args.binance_dir) if f.endswith(".csv")}
        gfiles = {f[:-4] for f in os.listdir(args.bitget_dir) if f.endswith(".csv")}
        symbols = sorted(list(bfiles.intersection(gfiles)))
        # Cap number of symbols as requested, hard upper bound at 100
        max_syms = max(0, min(int(args.max_symbols), 20))
        if max_syms and len(symbols) > max_syms:
            symbols = symbols[:max_syms]
    else:
        symbols = []

    if not symbols:
        print("No symbols found in data directories.")
        return

    # 1) Preprocess to BBO 1s and rolling stats, with memory cleanup between pairs
    ok_syms = preprocess_pairs(
        symbols,
        args.binance_dir,
        args.bitget_dir,
        args.bbo_dir,
        args.window,
        args.min_periods,
    )
    if not ok_syms:
        print("No symbols preprocessed successfully.")
        return

    # 2) Portfolio backtest over all processed pairs
    sparams = Params(
        profit_threshold=args.profit_threshold,
        fee_main=args.fee_main,
        fee_hedge=args.fee_hedge,
        close_mode=args.close_mode,
        window=args.window,
        min_periods=args.min_periods,
    )
    pparams = PortfolioParams(
        initial_capital=args.initial_capital,
        max_weight_per_pair=args.max_weight_per_pair,
        top_n=args.top_n,
    )

    trades_df, equity_df = run_portfolio(ok_syms, args.bbo_dir, pparams, sparams)

    # 3) Save results
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    all_trades_path = os.path.join(out_dir, "trades_all_pairs.csv")
    equity_path = os.path.join(out_dir, "equity_all_pairs.csv")
    trades_df.to_csv(all_trades_path, index=False)
    equity_df.to_csv(equity_path)

    metrics = compute_portfolio_metrics(trades_df, equity_df)
    compare_dir = os.path.join(out_dir, "compare")
    os.makedirs(compare_dir, exist_ok=True)
    out_path = os.path.join(compare_dir, "backtest_all_pairs_results.csv")
    row = metrics.copy()
    row["symbols"] = len(ok_syms)
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"Saved portfolio metrics to {out_path}")


if __name__ == "__main__":
    main()
