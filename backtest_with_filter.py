# -*- coding: utf-8 -*-
"""
Backtest with moving-average stability filter (Task 2 per AGENTS.md).

Adds a stability filter based on fast/slow EMAs (1h/4h by default) as in analysis_all.py:
- When unstable, do not open new positions.
- If a position is held when instability occurs (or persists), close immediately and wait until stability resumes.

Implements the same trading logic as backtest.py (entries/exits/fees/metrics),
with the additional MA-based filter.
"""

from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Data loading (same as backtest.py)
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
# Stability filter (streaming EMA logic like analysis_all.py v3)
# -----------------------------

@dataclass
class StabilityParams:
    fast_hours: float = 1.0
    slow_hours: float = 4.0
    min_periods: int = 600
    wait_minutes: int = 30
    enter_confirm_secs: int = 60
    std_floor: float = 1e-4
    enter_threshold_mult: float = 0.5
    reset_threshold_mult: float = 0.3
    stable_threshold_mult: float = 0.1
    stable_need_secs: int = 180


class _EMAStd:
    def __init__(self, alpha: float, min_periods: int):
        self.alpha = alpha
        self.min_periods = min_periods
        self.count = 0
        self.mean = 0.0
        self.sq_mean = 0.0
        self._buf: List[float] = []

    def update(self, x: float) -> Tuple[float, float]:
        self.count += 1
        if self.count <= self.min_periods:
            self._buf.append(x)
            m = float(np.mean(self._buf))
            v = float(np.var(self._buf))
            if self.count == self.min_periods:
                self.mean = m
                self.sq_mean = v + m * m
                self._buf.clear()
            return m, math.sqrt(v)
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.sq_mean = (1 - self.alpha) * self.sq_mean + self.alpha * (x * x)
        var = max(self.sq_mean - self.mean * self.mean, 0.0)
        return self.mean, math.sqrt(var)

    def reset(self):
        self.count = 0
        self.mean = 0.0
        self.sq_mean = 0.0
        self._buf.clear()


def compute_stability(series: pd.Series, sp: StabilityParams) -> pd.Series:
    alpha_fast = 2.0 / (sp.fast_hours * 3600 + 1)
    alpha_slow = 2.0 / (sp.slow_hours * 3600 + 1)
    fast = _EMAStd(alpha_fast, sp.min_periods)
    slow = _EMAStd(alpha_slow, sp.min_periods)

    unstable = False
    cooldown_until: Optional[pd.Timestamp] = None
    stable_secs = 0
    enter_breach_start: Optional[pd.Timestamp] = None
    wait_delta = pd.Timedelta(minutes=sp.wait_minutes)

    flags: List[bool] = []
    for ts, v in series.items():
        mf, _ = fast.update(float(v))
        ms, std_slow = slow.update(float(v))
        std_eff = max(std_slow, sp.std_floor)
        d = abs(mf - ms)

        if not unstable:
            if d > sp.enter_threshold_mult * std_eff:
                if enter_breach_start is None:
                    enter_breach_start = ts
                if (ts - enter_breach_start) >= pd.Timedelta(seconds=sp.enter_confirm_secs):
                    unstable = True
                    cooldown_until = ts + wait_delta
                    stable_secs = 0
                    enter_breach_start = None
                    fast.reset(); slow.reset()
                    fast.update(float(v)); slow.update(float(v))
            else:
                enter_breach_start = None
        else:
            if d > sp.reset_threshold_mult * std_eff:
                cooldown_until = ts + wait_delta
                stable_secs = 0
                fast.reset(); slow.reset()
                fast.update(float(v)); slow.update(float(v))
            else:
                if cooldown_until is not None and ts >= cooldown_until:
                    if d < sp.stable_threshold_mult * std_eff:
                        stable_secs += 1
                    else:
                        stable_secs = 0
                    if stable_secs >= sp.stable_need_secs:
                        unstable = False
                        cooldown_until = None
                        stable_secs = 0

        flags.append(unstable)

    return pd.Series(flags, index=series.index, name="is_unstable")


# -----------------------------
# Trading logic (same core as backtest.py)
# -----------------------------

CloseMode = Literal["reverse", "line", "both"]


@dataclass
class Params:
    profit_threshold: float = 0.0003
    fee_main: float = 0.0
    fee_hedge: float = 0.00015
    close_mode: CloseMode = "both"
    window: str = "4h"
    min_periods: int = 600


@dataclass
class Position:
    side: Literal["long", "short"]
    entry_time: pd.Timestamp
    entry_main_price: float
    entry_hedge_price: float


def estimate_entry_profit_margin(row: pd.Series, params: Params, side: Literal["long", "short"]) -> float:
    fee_buffer = 2.0 * params.fee_hedge
    if side == "long":
        margin = (row["bid_mean"] + row["bid_std"]) - row["bid_bid"]
    else:
        margin = row["ask_ask"] - (row["ask_mean"] - row["ask_std"])
    return float(margin - fee_buffer)


def should_open(row: pd.Series, params: Params, side: Literal["long", "short"]) -> bool:
    if not np.isfinite(row.get("bid_mean", np.nan)) or not np.isfinite(row.get("ask_mean", np.nan)):
        return False
    return estimate_entry_profit_margin(row, params, side) > params.profit_threshold


def check_close_signals(row: pd.Series, params: Params, side: Literal["long", "short"]) -> Tuple[bool, bool]:
    reverse_signal = should_open(row, params, "short" if side == "long" else "long")
    if side == "long":
        line_signal = (row["ask_ask"] >= (row["ask_mean"] + row["ask_std"])) if np.isfinite(row["ask_mean"]) else False
    else:
        line_signal = (row["bid_bid"] <= (row["bid_mean"] - row["bid_std"])) if np.isfinite(row["bid_mean"]) else False
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


def backtest_with_filter(
    df: pd.DataFrame,
    params: Params,
    stability: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trades: List[dict] = []
    equity_times: List[pd.Timestamp] = []
    equity_vals: List[float] = []

    pos: Optional[Position] = None
    equity = 1.0

    for ts, row in df.iterrows():
        is_unstable = bool(stability.loc[ts]) if ts in stability.index else True

        equity_times.append(ts)
        equity_vals.append(equity)

        if pos is not None and is_unstable:
            # Forced close due to instability
            if pos.side == "long":
                exit_main = float(row["ask_binance"])  # sell main at ask
                exit_hedge = float(row["ask_bitget"])  # buy hedge at ask
            else:
                exit_main = float(row["bid_binance"])  # buy main at bid
                exit_hedge = float(row["bid_bitget"])  # sell hedge at bid
            pnl, ret = execute_pnl(pos, ts, exit_main, exit_hedge, params)
            equity *= (1.0 + ret)
            trades.append({
                "side": pos.side,
                "entry_time": pos.entry_time,
                "exit_time": ts,
                "entry_main": pos.entry_main_price,
                "entry_hedge": pos.entry_hedge_price,
                "exit_main": exit_main,
                "exit_hedge": exit_hedge,
                "pnl": pnl,
                "ret": ret,
                "holding_minutes": float((ts - pos.entry_time).total_seconds() / 60.0),
                "reason": "unstable_close",
            })
            pos = None
            continue

        if is_unstable:
            # Skip trading while unstable
            continue

        if pos is None:
            open_long = should_open(row, params, "long")
            open_short = should_open(row, params, "short")
            if open_long or open_short:
                if open_long and not open_short:
                    side = "long"
                elif open_short and not open_long:
                    side = "short"
                else:
                    m_long = estimate_entry_profit_margin(row, params, "long")
                    m_short = estimate_entry_profit_margin(row, params, "short")
                    side = "long" if m_long >= m_short else "short"

                if side == "long":
                    entry_main = float(row["bid_binance"])  # buy main at bid
                    entry_hedge = float(row["bid_bitget"])  # sell hedge at bid
                else:
                    entry_main = float(row["ask_binance"])  # sell main at ask
                    entry_hedge = float(row["ask_bitget"])  # buy hedge at ask
                pos = Position(side=side, entry_time=ts, entry_main_price=entry_main, entry_hedge_price=entry_hedge)
        else:
            reverse_sig, line_sig = check_close_signals(row, params, pos.side)
            do_close = (params.close_mode == "reverse" and reverse_sig) or \
                       (params.close_mode == "line" and line_sig) or \
                       (params.close_mode == "both" and (reverse_sig or line_sig))
            if do_close:
                if pos.side == "long":
                    exit_main = float(row["ask_binance"])  # sell main at ask
                    exit_hedge = float(row["ask_bitget"])  # buy hedge at ask
                else:
                    exit_main = float(row["bid_binance"])  # buy main at bid
                    exit_hedge = float(row["bid_bitget"])  # sell hedge at bid
                pnl, ret = execute_pnl(pos, ts, exit_main, exit_hedge, params)
                equity *= (1.0 + ret)
                trades.append({
                    "side": pos.side,
                    "entry_time": pos.entry_time,
                    "exit_time": ts,
                    "entry_main": pos.entry_main_price,
                    "entry_hedge": pos.entry_hedge_price,
                    "exit_main": exit_main,
                    "exit_hedge": exit_hedge,
                    "pnl": pnl,
                    "ret": ret,
                    "holding_minutes": float((ts - pos.entry_time).total_seconds() / 60.0),
                    "reason": "signal_close",
                })
                pos = None

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame({"equity": equity_vals}, index=pd.to_datetime(equity_times))
    return trades_df, equity_df


# -----------------------------
# Metrics (aligned with backtest.py tweaks: no CAGR, avg holding minutes)
# -----------------------------

def compute_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
    metrics = {
        "trades": int(len(trades_df)),
        "wins": int((trades_df["ret"] > 0).sum()) if not trades_df.empty else 0,
        "win_rate": float(((trades_df["ret"] > 0).mean()) if not trades_df.empty else 0.0),
        "total_return": float((equity_df["equity"].iloc[-1] - equity_df["equity"].iloc[0]) if not equity_df.empty else 0.0),
        "total_ret_pct": float((equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1.0) if not equity_df.empty else 0.0),
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "duration_days": 0.0,
        "avg_holding_minutes": 0.0,
        "unstable_closes": int((trades_df.get("reason") == "unstable_close").sum()) if not trades_df.empty else 0,
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
        metrics["sharpe"] = float(daily.mean() / daily.std(ddof=0) * math.sqrt(365.0))
    else:
        metrics["sharpe"] = 0.0

    if not trades_df.empty:
        if "holding_minutes" in trades_df.columns:
            metrics["avg_holding_minutes"] = float(trades_df["holding_minutes"].mean())
        else:
            durations = (pd.to_datetime(trades_df["exit_time"]) - pd.to_datetime(trades_df["entry_time"]))
            metrics["avg_holding_minutes"] = float((durations.dt.total_seconds() / 60.0).mean()) if len(durations) else 0.0

    return metrics


# -----------------------------
# CLI entry
# -----------------------------

def run_for_symbol(symbol: str, args) -> Tuple[str, dict]:
    df = load_merged_orderbook(symbol, args.binance_dir, args.bitget_dir, resample="1s")
    df = add_spread_stats(df, window=args.window, min_periods=args.min_periods)

    # Stability uses bid/bid ratio as the value stream (consistent with analysis_all)
    ratio_series = (df["bid_binance"] / df["bid_bitget"]).rename("ratio")
    sp = StabilityParams(
        fast_hours=args.fast_hours,
        slow_hours=args.slow_hours,
        min_periods=args.stability_min_periods,
        wait_minutes=args.wait_minutes,
        enter_confirm_secs=args.enter_confirm_secs,
        std_floor=args.std_floor,
        enter_threshold_mult=args.enter_mult,
        reset_threshold_mult=args.reset_mult,
        stable_threshold_mult=args.stable_mult,
        stable_need_secs=args.stable_need_secs,
    )
    stability = compute_stability(ratio_series, sp)

    params = Params(
        profit_threshold=args.profit_threshold,
        fee_main=args.fee_main,
        fee_hedge=args.fee_hedge,
        close_mode=args.close_mode,
        window=args.window,
        min_periods=args.min_periods,
    )
    trades_df, equity_df = backtest_with_filter(df, params, stability)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    trades_path = os.path.join(out_dir, f"trades_filter_{symbol}.csv")
    equity_path = os.path.join(out_dir, f"equity_filter_{symbol}.csv")
    stability_path = os.path.join(out_dir, f"stability_{symbol}.csv")
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path)
    stability.to_frame().to_csv(stability_path)

    metrics = compute_metrics(trades_df, equity_df)
    return symbol, metrics


def main():
    parser = argparse.ArgumentParser(description="带均线过滤的价差套利回测")
    parser.add_argument("--symbols", nargs="+", default=["BNBUSDT"], help="交易对列表")
    parser.add_argument("--all", action="store_true", help="使用数据目录交集的所有交易对")
    parser.add_argument("--binance-dir", default="data/binance", help="Binance 数据目录")
    parser.add_argument("--bitget-dir", default="data/bitget", help="Bitget 数据目录")
    parser.add_argument("--output-dir", default="./data/backtest", help="输出目录")

    # Trading params (same as backtest.py)
    parser.add_argument("--window", default="4h", help="回归线滚动窗口，默认4h")
    parser.add_argument("--min-periods", type=int, default=600, help="滚动统计最小样本数")
    parser.add_argument("--profit-threshold", type=float, default=0.0003, help="开仓利润阈值（比率单位）")
    parser.add_argument("--fee-main", type=float, default=0.0, help="主交易所手续费")
    parser.add_argument("--fee-hedge", type=float, default=0.00015, help="对冲交易所手续费")
    parser.add_argument("--close-mode", choices=["reverse", "line", "both"], default="both", help="平仓方式")

    # Stability filter params (aligned with analysis_all.py v3)
    parser.add_argument("--fast-hours", type=float, default=1.0, help="快速EMA窗口（小时）")
    parser.add_argument("--slow-hours", type=float, default=4.0, help="慢速EMA窗口（小时）")
    parser.add_argument("--stability-min-periods", type=int, default=600, help="稳定性EMA初始化最小样本数")
    parser.add_argument("--wait-minutes", type=int, default=30, help="不稳定后等待分钟数")
    parser.add_argument("--enter-confirm-secs", type=int, default=60, help="进入不稳定需持续秒数")
    parser.add_argument("--std-floor", type=float, default=1e-4, help="标准差下限")
    parser.add_argument("--enter-mult", type=float, default=0.5, help="进入不稳定阈值乘数")
    parser.add_argument("--reset-mult", type=float, default=0.3, help="不稳定阶段重置阈值乘数")
    parser.add_argument("--stable-mult", type=float, default=0.1, help="恢复稳定阈值乘数")
    parser.add_argument("--stable-need-secs", type=int, default=180, help="恢复稳定需持续秒数")

    args = parser.parse_args()

    symbols = args.symbols
    if args.all:
        if os.path.isdir(args.binance_dir) and os.path.isdir(args.bitget_dir):
            bfiles = {f[:-4] for f in os.listdir(args.binance_dir) if f.endswith(".csv")}
            gfiles = {f[:-4] for f in os.listdir(args.bitget_dir) if f.endswith(".csv")}
            symbols = sorted(list(bfiles.intersection(gfiles)))

    results = []
    for sym in symbols:
        try:
            sym_, metrics = run_for_symbol(sym, args)
            results.append((sym_, metrics))
        except Exception as e:
            print(f"Error on {sym}: {e}")

    if results:
        print("Backtest With Filter Summary")
        for sym, m in results:
            print(
                f"{sym}: trades={m['trades']} win_rate={m['win_rate']:.2%} total_ret={m['total_ret_pct']:.2%} "
                f"MDD={m['max_drawdown']:.2%} Sharpe={m['sharpe']:.2f} days={m['duration_days']:.1f} "
                f"avg_hold={m['avg_holding_minutes']:.2f}m unstable_closes={m['unstable_closes']}"
            )


if __name__ == "__main__":
    main()

