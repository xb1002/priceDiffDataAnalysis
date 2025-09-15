# -*- coding: utf-8 -*-
"""
Run both backtests (with and without MA stability filter) using the same parameters,
and write results to CSV files per AGENTS.md Task 3:
  - backtest_results.csv
  - backtest_with_filter_results.csv

This script imports the two modules and drives their run_for_symbol functions with a
shared argument namespace so parameters match.
"""

from __future__ import annotations

import os
import argparse
from types import SimpleNamespace
from typing import List, Dict, Tuple

import pandas as pd

import backtest as bt
import backtest_with_filter as btf


def _collect_symbols(all_flag: bool, symbols: List[str], binance_dir: str, bitget_dir: str) -> List[str]:
    if not all_flag:
        return symbols
    if os.path.isdir(binance_dir) and os.path.isdir(bitget_dir):
        bfiles = {f[:-4] for f in os.listdir(binance_dir) if f.endswith(".csv")}
        gfiles = {f[:-4] for f in os.listdir(bitget_dir) if f.endswith(".csv")}
        return sorted(list(bfiles.intersection(gfiles)))
    return symbols


def run_compare(
    symbols: List[str],
    binance_dir: str = "data/binance",
    bitget_dir: str = "data/bitget",
    output_dir: str = "./data/backtest",
    window: str = "4h",
    min_periods: int = 600,
    profit_threshold: float = 0.0003,
    fee_main: float = 0.0,
    fee_hedge: float = 0.00015,
    close_mode: str = "both",
    # Stability params
    fast_hours: float = 1.0,
    slow_hours: float = 4.0,
    stability_min_periods: int = 600,
    wait_minutes: int = 30,
    enter_confirm_secs: int = 60,
    std_floor: float = 1e-4,
    enter_mult: float = 0.5,
    reset_mult: float = 0.3,
    stable_mult: float = 0.1,
    stable_need_secs: int = 180,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    compare_dir = os.path.join(output_dir, "compare")
    os.makedirs(compare_dir, exist_ok=True)

    # Build arg namespaces compatible with bt.run_for_symbol and btf.run_for_symbol
    common = SimpleNamespace(
        binance_dir=binance_dir,
        bitget_dir=bitget_dir,
        output_dir=output_dir,
        window=window,
        min_periods=min_periods,
        profit_threshold=profit_threshold,
        fee_main=fee_main,
        fee_hedge=fee_hedge,
        close_mode=close_mode,
    )

    filter_args = SimpleNamespace(
        **common.__dict__,
        fast_hours=fast_hours,
        slow_hours=slow_hours,
        stability_min_periods=stability_min_periods,
        wait_minutes=wait_minutes,
        enter_confirm_secs=enter_confirm_secs,
        std_floor=std_floor,
        enter_mult=enter_mult,
        reset_mult=reset_mult,
        stable_mult=stable_mult,
        stable_need_secs=stable_need_secs,
    )

    rows_bt: List[Dict] = []
    rows_btf: List[Dict] = []

    for sym in symbols:
        try:
            _, m1 = bt.run_for_symbol(sym, common)
            row1 = {"symbol": sym}
            row1.update(m1)
            rows_bt.append(row1)
            # 打印结果
            print(
                f"{sym}: normal: trades={m1['trades']} win_rate={m1['win_rate']:.2%} total_ret={m1['total_ret_pct']:.2%} "
                f"MDD={m1['max_drawdown']:.2%} Sharpe={m1['sharpe']:.2f} days={m1['duration_days']:.1f} "
                f"avg_hold={m1['avg_holding_minutes']:.2f}m"
            )
        except Exception as e:
            print(f"[backtest] Error on {sym}: {e}")

        try:
            _, m2 = btf.run_for_symbol(sym, filter_args)
            row2 = {"symbol": sym}
            row2.update(m2)
            rows_btf.append(row2)
            # 打印结果
            print(
                f"{sym}: filtered: trades={m2['trades']} win_rate={m2['win_rate']:.2%} total_ret={m2['total_ret_pct']:.2%} "
                f"MDD={m2['max_drawdown']:.2%} Sharpe={m2['sharpe']:.2f} days={m2['duration_days']:.1f} "
                f"avg_hold={m2['avg_holding_minutes']:.2f}m unstable_closes={m2['unstable_closes']}"
            )
        except Exception as e:
            print(f"[backtest_with_filter] Error on {sym}: {e}")

    df_bt = pd.DataFrame(rows_bt)
    df_btf = pd.DataFrame(rows_btf)

    bt_csv = os.path.join(compare_dir, "backtest_results.csv")
    btf_csv = os.path.join(compare_dir, "backtest_with_filter_results.csv")
    if not df_bt.empty:
        df_bt.to_csv(bt_csv, index=False)
        print(f"Saved metrics to {bt_csv}")
    if not df_btf.empty:
        df_btf.to_csv(btf_csv, index=False)
        print(f"Saved metrics to {btf_csv}")

    return df_bt, df_btf


def main():
    parser = argparse.ArgumentParser(description="对比回测：普通 vs 带均线过滤")
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
    parser.add_argument("--close-mode", choices=["reverse", "line", "both"], default="both", help="平仓方式")

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

    syms = _collect_symbols(args.all, args.symbols, args.binance_dir, args.bitget_dir)
    run_compare(
        symbols=syms,
        binance_dir=args.binance_dir,
        bitget_dir=args.bitget_dir,
        output_dir=args.output_dir,
        window=args.window,
        min_periods=args.min_periods,
        profit_threshold=args.profit_threshold,
        fee_main=args.fee_main,
        fee_hedge=args.fee_hedge,
        close_mode=args.close_mode,
        fast_hours=args.fast_hours,
        slow_hours=args.slow_hours,
        stability_min_periods=args.stability_min_periods,
        wait_minutes=args.wait_minutes,
        enter_confirm_secs=args.enter_confirm_secs,
        std_floor=args.std_floor,
        enter_mult=args.enter_mult,
        reset_mult=args.reset_mult,
        stable_mult=args.stable_mult,
        stable_need_secs=args.stable_need_secs,
    )


if __name__ == "__main__":
    main()
