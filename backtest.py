# -*- coding: utf-8 -*-
"""
Backtest for spread arbitrage per AGENTS.md

Key rules:
- Use 4-hour moving average as regression line; track moving mean and std.
- Two exchanges: main and hedge (default: Binance main, Bitget hedge based on data dirs).
- Two directions:
  - Long: buy on main at main bid; sell on hedge at hedge bid.
  - Short: sell on main at main ask; buy on hedge at hedge ask.
- Close lines:
  - Long closes when ask/ask ratio >= ask_mean + std, or by reverse (Short) signal.
  - Short closes when bid/bid ratio <= bid_mean - std, or by reverse (Long) signal.
- Open only if profit to close-line exceeds threshold.
- Fees: main fee 0.00%, hedge fee 0.015% (applied on hedge legs at open and close).
- Report total return, CAGR, max drawdown, Sharpe, win rate.

CLI example:
  python backtest.py --symbols BNBUSDT AAVEUSDT --profit-threshold 0.001 --close-mode both
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
# Data loading and preparation
# -----------------------------

def load_merged_orderbook(
    symbol: str,
    binance_dir: str = "data/binance",
    bitget_dir: str = "data/bitget",
    resample: str = "1s",
) -> pd.DataFrame:
    """Load orderbook best bid/ask for Binance (main) and Bitget (hedge),
    align to 1 second, forward-fill, and return merged DataFrame with columns:
      - bid_binance, ask_binance, bid_bitget, ask_bitget
    Index is pandas datetime index.
    """
    binance_fp = os.path.join(binance_dir, f"{symbol}.csv")
    bitget_fp = os.path.join(bitget_dir, f"{symbol}.csv")

    if not (os.path.exists(binance_fp) and os.path.exists(bitget_fp)):
        raise FileNotFoundError(f"Missing CSV for {symbol}: {binance_fp} or {bitget_fp}")

    # Binance file expected columns: b, a, T (ms)
    # Bitget file expected columns: bidPr, askPr, ts (ms)
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

    # Clean duplicates and NaNs, sort
    bdf = bdf[~bdf.index.duplicated(keep="first")].sort_index()
    gdf = gdf[~gdf.index.duplicated(keep="first")].sort_index()
    bdf = bdf[bdf.index.notna()].dropna()
    gdf = gdf[gdf.index.notna()].dropna()

    # Align to 1-second frequency
    b1 = bdf.resample(resample).ffill().dropna()
    g1 = gdf.resample(resample).ffill().dropna()

    merged = b1.merge(g1, left_index=True, right_index=True, suffixes=("_binance", "_bitget"))
    # Ensure positive prices
    merged = merged[(merged[["bid_binance", "ask_binance", "bid_bitget", "ask_bitget"]] > 0).all(axis=1)]
    return merged


def check_data_coverage(df: pd.DataFrame, threshold: float = 0.70) -> Tuple[bool, float, int, int]:
    """Check 1-second data coverage between first and last timestamps.
    Returns (ok, coverage, actual_rows, expected_seconds).
    """
    if df is None or df.empty:
        return False, 0.0, 0, 0
    start, end = df.index[0], df.index[-1]
    total_secs = int((end - start).total_seconds()) + 1
    if total_secs <= 0:
        return False, 0.0, len(df), max(total_secs, 0)
    actual = int(len(df))
    coverage = actual / float(total_secs)
    return coverage >= threshold, float(coverage), actual, total_secs


def add_spread_stats(
    df: pd.DataFrame,
    window: str = "4h",
    min_periods: int = 600,
) -> pd.DataFrame:
    """Add bid/bid and ask/ask ratios and their rolling mean/std using time-based window.
    Columns added:
      - bid_bid, ask_ask
      - bid_mean, bid_std, ask_mean, ask_std
    """
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
# Strategy and backtest
# -----------------------------

CloseMode = Literal["reverse", "line", "both"]


@dataclass
class Params:
    profit_threshold: float = 0.0004  # minimum expected ratio distance to close-line to open
    fee_main: float = 0.0
    fee_hedge: float = 0.00015  # 0.015%
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
    """Return the expected profit to the close-line in ratio-units, net of a simple fee buffer.

    Per spec:
      - Long margin = (bid_mean + bid_std - bid_bid_current)
      - Short margin = (ask_ask_current - (ask_mean - ask_std))
    Subtract 2*fee_hedge as a conservative buffer for roundtrip hedge fees.
    """
    fee_buffer = 2.0 * params.fee_hedge
    if side == "long":
        margin = (row["ask_mean"] + row["ask_std"]) - row["bid_bid"]
    else:  # short
        margin = row["ask_ask"] - (row["bid_mean"] - row["bid_std"])
    return float(margin - fee_buffer)


def should_open(row: pd.Series, params: Params, side: Literal["long", "short"]) -> bool:
    # Require stats ready
    if not np.isfinite(row.get("bid_mean", np.nan)) or not np.isfinite(row.get("ask_mean", np.nan)):
        return False
    margin = estimate_entry_profit_margin(row, params, side)
    return margin > params.profit_threshold


def check_close_signals(row: pd.Series, params: Params, side: Literal["long", "short"]) -> Tuple[bool, bool]:
    """Return tuple (reverse_signal, line_signal) for the given position side.

    - Reverse signal is simply the opposite side's open signal.
    - Line signal:
        * For long: ask_ask >= ask_mean + ask_std
        * For short: bid_bid <= bid_mean - bid_std
    """
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
    """Compute PnL in quote currency and return fraction relative to entry main price for 1 unit size.

    Long roundtrip cash flow (size = 1 unit of asset):
      open:  -Pm (buy main at bid) + Ph*(1 - fh) (sell hedge at bid)
      close: +Qm (sell main at ask) - Qh*(1 + fh) (buy hedge at ask)
      PnL = (Ph*(1 - fh) - Pm) + (Qm - Qh*(1 + fh))

    Short roundtrip:
      open:  +Pm (sell main at ask) - Ph*(1 + fh) (buy hedge at ask)
      close: -Qm (buy main at bid) + Qh*(1 - fh) (sell hedge at bid)
      PnL = (Pm - Ph*(1 + fh)) + (Qh*(1 - fh) - Qm)
    """
    fh = params.fee_hedge
    if pos.side == "long":
        Pm = pos.entry_main_price
        Ph = pos.entry_hedge_price
        Qm = exit_main_price
        Qh = exit_hedge_price
        pnl = (Ph * (1 - fh) - Pm) + (Qm - Qh * (1 + fh))
        ret = pnl / max(Pm, 1e-12)
    else:  # short
        Pm = pos.entry_main_price
        Ph = pos.entry_hedge_price
        Qm = exit_main_price
        Qh = exit_hedge_price
        pnl = (Pm - Ph * (1 + fh)) + (Qh * (1 - fh) - Qm)
        ret = pnl / max(Pm, 1e-12)
    return pnl, ret


def backtest_one(
    df: pd.DataFrame,
    params: Params,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run backtest on prepared DataFrame with spread stats.

    Returns (trades_df, equity_df)
    """
    trades: List[dict] = []
    equity_times: List[pd.Timestamp] = []
    equity_vals: List[float] = []

    pos: Optional[Position] = None
    equity = 1.0  # start equity for compounded calculation

    for ts, row in df.iterrows():
        # Capture equity at each bar end for curve (stepwise)
        equity_times.append(ts)
        equity_vals.append(equity)

        if pos is None:
            # Try open Long or Short; prefer the bigger margin if both qualify
            open_long = should_open(row, params, "long")
            open_short = should_open(row, params, "short")
            if open_long or open_short:
                # Prices per spec
                if open_long and not open_short:
                    side = "long"
                elif open_short and not open_long:
                    side = "short"
                else:
                    # both true: choose larger margin
                    m_long = estimate_entry_profit_margin(row, params, "long")
                    m_short = estimate_entry_profit_margin(row, params, "short")
                    side = "long" if m_long >= m_short else "short"

                if side == "long":
                    entry_main = float(row["bid_binance"])  # buy on main at bid
                    entry_hedge = float(row["bid_bitget"])  # sell on hedge at bid
                else:  # short
                    entry_main = float(row["ask_binance"])  # sell on main at ask
                    entry_hedge = float(row["ask_bitget"])  # buy on hedge at ask

                pos = Position(side=side, entry_time=ts, entry_main_price=entry_main, entry_hedge_price=entry_hedge)
        else:
            # Check close signals for current position
            reverse_sig, line_sig = check_close_signals(row, params, pos.side)

            do_close = False
            if params.close_mode == "reverse":
                do_close = reverse_sig
            elif params.close_mode == "line":
                do_close = line_sig
            else:  # both: close if either
                do_close = reverse_sig or line_sig

            if do_close:
                if pos.side == "long":
                    exit_main = float(row["ask_binance"])  # sell main at ask
                    exit_hedge = float(row["ask_bitget"])  # buy hedge at ask
                else:  # short
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
                })
                pos = None

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame({"equity": equity_vals}, index=pd.to_datetime(equity_times))
    return trades_df, equity_df


# -----------------------------
# Metrics
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
    }
    if equity_df.empty:
        return metrics

    eq = equity_df["equity"].astype(float)
    start, end = eq.index[0], eq.index[-1]
    days = max((end - start).total_seconds() / (3600 * 24), 1e-12)
    metrics["duration_days"] = days

    # Max drawdown
    roll_max = eq.cummax()
    drawdown = (eq - roll_max) / roll_max
    metrics["max_drawdown"] = float(drawdown.min())

    # Sharpe: use daily returns from equity curve
    daily = eq.resample("1D").last().pct_change().dropna()
    if len(daily) > 1 and daily.std(ddof=0) > 0:
        sharpe = float(daily.mean() / daily.std(ddof=0) * math.sqrt(365.0))
        metrics["sharpe"] = sharpe
    else:
        metrics["sharpe"] = 0.0

    # Average holding time
    if not trades_df.empty:
        if "holding_minutes" in trades_df.columns:
            avg_min = float(trades_df["holding_minutes"].mean())
        else:
            # fallback compute from timestamps
            durations = (pd.to_datetime(trades_df["exit_time"]) - pd.to_datetime(trades_df["entry_time"]))
            avg_min = float((durations.dt.total_seconds() / 60.0).mean()) if len(durations) else 0.0
        metrics["avg_holding_minutes"] = avg_min

    return metrics


# -----------------------------
# CLI entry
# -----------------------------

def run_for_symbol(symbol: str, args) -> Tuple[str, dict]:
    df = load_merged_orderbook(symbol, args.binance_dir, args.bitget_dir, resample="1s")
    ok, cov, actual, total = check_data_coverage(df, threshold=0.70)
    if not ok:
        print(f"Skipping {symbol}: data coverage {cov:.2%} ({actual}/{total}) < 70%")
        # Return empty metrics to signal skip
        return symbol, compute_metrics(pd.DataFrame(), pd.DataFrame())
    df = add_spread_stats(df, window=args.window, min_periods=args.min_periods)

    params = Params(
        profit_threshold=args.profit_threshold,
        fee_main=args.fee_main,
        fee_hedge=args.fee_hedge,
        close_mode=args.close_mode,
        window=args.window,
        min_periods=args.min_periods,
    )
    trades_df, equity_df = backtest_one(df, params)

    # Save trades
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    trades_path = os.path.join(out_dir, f"trades_{symbol}.csv")
    equity_path = os.path.join(out_dir, f"equity_{symbol}.csv")
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path)

    metrics = compute_metrics(trades_df, equity_df)
    return symbol, metrics


def main():
    parser = argparse.ArgumentParser(description="价差套利策略回测 (4小时均线回归线)")
    parser.add_argument("--symbols", nargs="+", default=["BNBUSDT"], help="交易对列表")
    parser.add_argument("--all", action="store_true", help="使用数据目录交集的所有交易对")
    parser.add_argument("--binance-dir", default="data/binance", help="Binance 数据目录")
    parser.add_argument("--bitget-dir", default="data/bitget", help="Bitget 数据目录")
    parser.add_argument("--output-dir", default="./data/backtest", help="输出目录")

    parser.add_argument("--window", default="4h", help="回归线滚动窗口，默认4h")
    parser.add_argument("--min-periods", type=int, default=600, help="滚动统计最小样本数")

    parser.add_argument("--profit-threshold", type=float, default=0.0003, help="开仓利润阈值（比率单位）")
    parser.add_argument("--fee-main", type=float, default=0.0, help="主交易所手续费")
    parser.add_argument("--fee-hedge", type=float, default=0.00015, help="对冲交易所手续费")
    parser.add_argument(
        "--close-mode",
        choices=["reverse", "line", "both"],
        default="both",
        help="平仓方式：反向、到线、二者其一",
    )

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
            print(f"{sym_}: trades={metrics['trades']} win_rate={metrics['win_rate']:.2%} "
                  f"total_ret={metrics['total_ret_pct']:.2%} MDD={metrics['max_drawdown']:.2%} "
                  f"Sharpe={metrics['sharpe']:.2f} days={metrics['duration_days']:.1f} "
                  f"avg_hold={metrics['avg_holding_minutes']:.2f}m")
        except Exception as e:
            print(f"Error on {sym}: {e}")

    # Write aggregated metrics to CSV
    if results:
        # Save aggregated metrics under <output_dir>/compare/
        compare_dir = os.path.join(args.output_dir, "compare")
        os.makedirs(compare_dir, exist_ok=True)
        out_path = os.path.join(compare_dir, "backtest_results.csv")
        rows = []
        for sym, m in results:
            row = {"symbol": sym}
            row.update(m)
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
