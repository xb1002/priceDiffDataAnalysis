# -*- coding: utf-8 -*-
"""
全币种回测（带均线稳定性过滤）

组合层面的价差套利回测，融合 backtest.py 的交易规则与 backtest_with_filter.py 的均线稳定性过滤：
- 预处理：对每个交易对生成 1s 的 BBO，并计算滚动均值与标准差，保存到 data/bbo/
- 稳定性过滤：基于快/慢 EMA 的差异判断不稳定期；不稳定期不新开仓，若持仓则立即平仓
- 资金分配：初始资金、单品种最大权重、每秒选择预期收益最高的前 N 个品种
- 输出：交易明细、组合权益曲线、汇总指标 CSV
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
# 数据加载与统计（与 backtest.py 对齐）
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
# 策略原语（与 backtest.py/backtest_with_filter.py 对齐）
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
    weight: float


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
# 稳定性过滤（与 backtest_with_filter.py 对齐）
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
# 预处理（与 backtest_all_pairs.py 对齐）
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
            cols = [
                "bid_binance", "ask_binance", "bid_bitget", "ask_bitget",
                "bid_bid", "ask_ask", "bid_mean", "bid_std", "ask_mean", "ask_std"
            ]
            out = df[cols].copy()
            out_path = os.path.join(bbo_dir, f"{sym}_bbo_1s.csv")
            out.to_csv(out_path)
            ok_syms.append(sym)
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
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


# -----------------------------
# 组合回测（带稳定性过滤）
# -----------------------------

@dataclass
class PortfolioParams:
    initial_capital: float = 10000.0
    max_weight_per_pair: float = 0.2
    top_n: int = 3


def compute_portfolio_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
    metrics = {
        "trades": int(len(trades_df)),
        "wins": int((trades_df.get("weighted_ret", pd.Series(dtype=float)) > 0).sum()) if not trades_df.empty else 0,
        "win_rate": float(((trades_df.get("weighted_ret", pd.Series(dtype=float)) > 0).mean()) if not trades_df.empty else 0.0),
        "unstable_closes": int((trades_df.get("reason", pd.Series(dtype=object)) == "unstable_close").sum()) if not trades_df.empty else 0,
        "total_return": float((equity_df["equity"].iloc[-1] - equity_df["equity"].iloc[0]) if not equity_df.empty else 0.0),
        "total_ret_pct": float((equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1.0) if not equity_df.empty else 0.0),
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "duration_days": 0.0,
        "avg_holding_minutes": float(trades_df.get("holding_minutes", pd.Series(dtype=float)).mean()) if not trades_df.empty else 0.0,
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


def run_portfolio_with_filter(
    symbols: List[str],
    bbo_dir: str,
    pparams: PortfolioParams,
    sparams: Params,
    stab_params: StabilityParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    series: Dict[str, pd.DataFrame] = {}
    stability: Dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            df = load_bbo(sym, bbo_dir)
            series[sym] = df
            ratio = (df["bid_binance"] / df["bid_bitget"]).rename("ratio")
            stability[sym] = compute_stability(ratio, stab_params)
        except Exception as e:
            print(f"Load skip {sym}: {e}")

    if not series:
        return pd.DataFrame(), pd.DataFrame()

    # Overlap time window
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

        # 1) Close conditions and forced close on instability
        to_close: List[str] = []
        for sym, pos in open_positions.items():
            df = series[sym]
            st = stability[sym]
            # If timestamp missing, treat as unstable to be conservative
            is_unstable = True
            if ts in st.index:
                is_unstable = bool(st.loc[ts])

            row = df.loc[ts]
            if is_unstable:
                # Forced close
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
                    "reason": "unstable_close",
                })
                used_weight -= pos.weight
                used_weight = max(used_weight, 0.0)
                to_close.append(sym)
                continue

            # Normal signal-based close if stable
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
                    "reason": "signal_close",
                })
                used_weight -= pos.weight
                used_weight = max(used_weight, 0.0)
                to_close.append(sym)

        for sym in to_close:
            del open_positions[sym]

        # 2) Open new positions only if capacity and stability is good
        capacity = max(1.0 - used_weight, 0.0)
        if capacity > 1e-9:
            candidates: List[Tuple[str, float, str]] = []  # (symbol, margin, side)
            for sym, df in series.items():
                if sym in open_positions:
                    continue
                st = stability[sym]
                if ts not in st.index or bool(st.loc[ts]):
                    # Missing or unstable -> skip
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
    parser = argparse.ArgumentParser(description="全币种回测（带均线过滤）")
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

    # Stability filter params
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

    # Discover symbols
    if os.path.isdir(args.binance_dir) and os.path.isdir(args.bitget_dir):
        bfiles = {f[:-4] for f in os.listdir(args.binance_dir) if f.endswith(".csv")}
        gfiles = {f[:-4] for f in os.listdir(args.bitget_dir) if f.endswith(".csv")}
        symbols = sorted(list(bfiles.intersection(gfiles)))
        max_syms = max(0, min(int(args.max_symbols), 20))
        if max_syms and len(symbols) > max_syms:
            symbols = symbols[:max_syms]
    else:
        symbols = []

    if not symbols:
        print("No symbols found in data directories.")
        return

    # 1) Preprocess per-pair to BBO with rolling stats
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

    # 2) Build params
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
    stab_params = StabilityParams(
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

    # 3) Run portfolio with filter
    trades_df, equity_df = run_portfolio_with_filter(ok_syms, args.bbo_dir, pparams, sparams, stab_params)

    # 4) Save outputs
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    trades_path = os.path.join(out_dir, "trades_all_pairs_filter.csv")
    equity_path = os.path.join(out_dir, "equity_all_pairs_filter.csv")
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path)

    metrics = compute_portfolio_metrics(trades_df, equity_df)
    compare_dir = os.path.join(out_dir, "compare")
    os.makedirs(compare_dir, exist_ok=True)
    out_path = os.path.join(compare_dir, "backtest_all_pairs_with_filter_results.csv")
    row = metrics.copy()
    row["symbols"] = len(ok_syms)
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"Saved portfolio metrics to {out_path}")


if __name__ == "__main__":
    main()

