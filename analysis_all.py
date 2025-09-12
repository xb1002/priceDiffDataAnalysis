import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# 显示中文
# Matplotlib 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 移动均值和标准差计算器
# 在线均值+方差计算器（与analysis.ipynb一致风格）
class MovingMeanStdCalculator:
    def __init__(self, alpha: float, min_periods: int = 1):
        self.alpha = alpha
        self.min_periods = min_periods
        self.count = 0
        self.mean = 0.0
        self.squared_mean = 0.0
        self.initial_values = []

    def update(self, value: float):
        self.count += 1
        # min_periods期间先用简单均值/方差
        if self.count <= self.min_periods:
            self.initial_values.append(value)
            simple_mean = sum(self.initial_values) / len(self.initial_values)
            if len(self.initial_values) == 1:
                simple_variance = 0.0
            else:
                simple_variance = sum((x - simple_mean) ** 2 for x in self.initial_values) / len(self.initial_values)
            simple_std = simple_variance ** 0.5
            if self.count == self.min_periods:
                self.mean = simple_mean
                self.squared_mean = simple_variance + simple_mean ** 2
                self.initial_values = []
            return simple_mean, simple_std
        # 之后用指数更新
        self.mean = (1 - self.alpha) * self.mean + self.alpha * value
        self.squared_mean = (1 - self.alpha) * self.squared_mean + self.alpha * (value * value)
        variance = self.squared_mean - (self.mean * self.mean)
        std_dev = (variance if variance >= 0 else 0) ** 0.5
        return self.mean, std_dev

    def reset(self):
        self.count = 0
        self.mean = 0.0
        self.squared_mean = 0.0
        self.initial_values = []

    def reset_moving_value(self, new_mean: float, new_variance: float = 0.0):
        self.mean = float(new_mean)
        self.squared_mean = float(new_variance + new_mean * new_mean)

    def variance(self):
        return self.squared_mean - (self.mean * self.mean)
    
# 数据加载
def get_ratio_data(
    symbol: str,
    binance_data_dir: str = 'data/binance/',
    bitget_data_dir: str = 'data/bitget/'
) -> pd.Series | None:

    binance_file_path = os.path.join(binance_data_dir, f'{symbol}.csv')
    bitget_file_path = os.path.join(bitget_data_dir, f'{symbol}.csv')

    try:
        # 读取原始CSV
        binance_df = pd.read_csv(binance_file_path)
        bitget_df = pd.read_csv(bitget_file_path)

        # 规范列名与索引：Binance(b,a,T ms) -> bid,ask,timestamp
        binance_df = binance_df[['b','a','T']].copy()
        binance_df.columns = ['bid','ask','timestamp']
        binance_df['timestamp'] = pd.to_datetime(binance_df['timestamp'], unit='ms')
        binance_df.set_index('timestamp', inplace=True)
        # 规范列名与索引：Bitget(bidPr,askPr,ts ms) -> bid,ask,timestamp
        bitget_df = bitget_df[['bidPr','askPr','ts']].copy()
        bitget_df.columns = ['bid','ask','timestamp']
        bitget_df['timestamp'] = pd.to_datetime(bitget_df['timestamp'], unit='ms')
        bitget_df.set_index('timestamp', inplace=True)

        # 清洗与对齐到1秒
        binance_df = binance_df[~binance_df.index.duplicated(keep='first')].sort_index()
        bitget_df = bitget_df[~bitget_df.index.duplicated(keep='first')].sort_index()
        binance_df = binance_df[binance_df.index.notna()]
        bitget_df = bitget_df[bitget_df.index.notna()]
        binance_1s = binance_df.resample('1s').ffill().dropna()
        bitget_1s = bitget_df.resample('1s').ffill().dropna()
        merged_1s = binance_1s.merge(bitget_1s, left_index=True, right_index=True, suffixes=('_binance','_bitget'))
        # 价差比：用bid对比
        merged_1s['bid_ratio'] = merged_1s['bid_binance'] / merged_1s['bid_bitget']
        ratio = merged_1s['bid_ratio']
        return ratio
    except Exception as e:
        print(f"Error loading data for {symbol}: {e}")
        return None
    
# 新稳定性检测：0.5×std连续≥60s触发不稳定->重置并等待30分钟；不稳定阶段若>0.3×std再次重置；
# 直到<0.1×std连续3分钟稳定
def stability_detection_v2(ratio_series: pd.Series, 
                           alpha_fast: float = None, alpha_slow: float = None, 
                           min_periods: int = 600, wait_minutes: int = 30, 
                           enter_confirm_secs: int = 60, 
                           std_floor: float = 1e-4):
    if alpha_fast is None:
        alpha_fast = 2/(1*60*60+1)  # 快速均线（短期）
    if alpha_slow is None:
        alpha_slow = 2/(4*60*60+1)  # 慢速均线（长期）
    
    # 均线（可重置）
    ma_fast_calc = MovingMeanStdCalculator(alpha=alpha_fast, min_periods=min_periods)
    ma_slow_calc = MovingMeanStdCalculator(alpha=alpha_slow, min_periods=min_periods)
    # 阈值所用的慢速标准差（不随重置清零，作为波动基线）
    std_slow_base = MovingMeanStdCalculator(alpha=alpha_slow, min_periods=min_periods)
    
    state_unstable = False
    unstable_start = None
    cooldown_until = None
    stable_secs = 0  # <0.1×std的累积秒数
    stable_need = 3*60  # 3分钟
    wait_delta = pd.Timedelta(minutes=wait_minutes)
    enter_breach_start = None  # 进入不稳定阈值连续计时起点
    
    results = []
    resets = []  # 记录每次重置事件
    
    print('开始稳定性检测...')
    for ts, v in ratio_series.items():
        ma_fast, _ = ma_fast_calc.update(v)
        ma_slow, std_slow = ma_slow_calc.update(v)
        # ma_slow, _ = ma_slow_calc.update(v)
        # _, std_slow = std_slow_base.update(v)
        std_slow_eff = max(std_slow, std_floor)
        d = abs(ma_fast - ma_slow)
        
        if not state_unstable:
            # 进入不稳定：|MA差| > 0.5×慢速std 且持续≥enter_confirm_secs
            if d > 0.5 * std_slow_eff:
                if enter_breach_start is None:
                    enter_breach_start = ts
                duration_ok = (ts - enter_breach_start) >= pd.Timedelta(seconds=enter_confirm_secs)
                if duration_ok:
                    state_unstable = True
                    unstable_start = ts
                    cooldown_until = ts + wait_delta
                    stable_secs = 0
                    enter_breach_start = None
                    # 重置两条均线（以当前值启动）
                    ma_fast_calc.reset(); ma_slow_calc.reset()
                    ma_fast_calc.update(v); ma_slow_calc.update(v)
                    print(f'不稳定开始: {ts}, 超过0.5×std已持续≥{enter_confirm_secs}s，阈值(0.5×)= {0.5*std_slow_eff:.6f}，进入等待{wait_minutes}分钟')
                    resets.append((ts, 'start'))
            else:
                # 未持续满足阈值，重置进入不稳定的计时
                enter_breach_start = None
        else:
            # 不稳定阶段：若差距>0.3×慢速std，则立即再次重置并重新等待30分钟
            if d > 0.3 * std_slow_eff:
                cooldown_until = ts + wait_delta
                stable_secs = 0
                ma_fast_calc.reset(); ma_slow_calc.reset()
                ma_fast_calc.update(v); ma_slow_calc.update(v)
                print(f'再次重置: {ts}, 差距={d:.6f} > 0.3×std，重新等待30分钟')
                resets.append((ts, 'reset'))
            else:
                # 到达等待期后才开始计稳定的3分钟
                if cooldown_until is not None and ts >= cooldown_until:
                    if d < 0.1 * std_slow_eff:
                        stable_secs += 1  # 假设数据为1秒采样
                    else:
                        stable_secs = 0
                    if stable_secs >= stable_need:
                        # 结束不稳定阶段
                        state_unstable = False
                        print(f'不稳定结束: {ts}, 已低于0.1×std持续3分钟')
                        unstable_start = None
                        cooldown_until = None
                        stable_secs = 0
        
        results.append({
            'timestamp': ts,
            'value': v,
            'ma_fast': ma_fast,
            'ma_slow': ma_slow,
            'std_slow_base': std_slow,
            'ma_diff': d,
            'is_unstable': state_unstable,
            'cooldown_until': cooldown_until,
        })
    
    print('稳定性检测完成')
    return pd.DataFrame(results), resets

def plot_results(symbol: str, result_df: pd.DataFrame, output_dir: str = "./data/images"):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 绘图：与原分析风格保持一致
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
    
    # 主图：比值与均线
    result_df['value'].plot(ax=ax1, color='tab:gray', alpha=0.5, label='bid/bid', linewidth=1)
    result_df['ma_fast'].plot(ax=ax1, color='tab:blue', label='快速均线', linewidth=2)
    result_df['ma_slow'].plot(ax=ax1, color='tab:green', label='慢速均线', linewidth=2)
    
    # 标记不稳定区间
    unstable_mask = result_df['is_unstable'].fillna(False).values
    unstable_starts, unstable_ends = [], []
    in_u = False
    idx = result_df.index
    for i, flag in enumerate(unstable_mask):
        if flag and not in_u:
            unstable_starts.append(idx[i]); in_u = True
        elif not flag and in_u:
            unstable_ends.append(idx[i-1] if i>0 else idx[i]); in_u = False
    if in_u:
        unstable_ends.append(idx[-1])
    
    for s, e in zip(unstable_starts, unstable_ends):
        ax1.axvspan(s, e, alpha=0.2, color='red', label='不稳定区间' if s==unstable_starts[0] else '')
    
    ax1.set_title('稳定性检测（新逻辑）- 价格比值与均线')
    ax1.set_ylabel('比值')
    ax1.set_xlabel('时间')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    filename = f"stability_analysis_{symbol}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存至: {filepath}")
    


# task
def stability_detection_v3(
    ratio_series: pd.Series,
    alpha_fast: float | None = None,
    alpha_slow: float | None = None,
    min_periods: int = 600,
    wait_minutes: int = 30,
    enter_confirm_secs: int = 60,
    std_floor: float = 1e-4,
    enter_threshold_mult: float = 0.5,
    reset_threshold_mult: float = 0.3,
    stable_threshold_mult: float = 0.1,
    stable_need_secs: int = 180,
):
    if alpha_fast is None:
        alpha_fast = 2 / (1 * 60 * 60 + 1)
    if alpha_slow is None:
        alpha_slow = 2 / (4 * 60 * 60 + 1)

    ma_fast_calc = MovingMeanStdCalculator(alpha=alpha_fast, min_periods=min_periods)
    ma_slow_calc = MovingMeanStdCalculator(alpha=alpha_slow, min_periods=min_periods)

    state_unstable = False
    cooldown_until = None
    stable_secs = 0
    wait_delta = pd.Timedelta(minutes=wait_minutes)
    enter_breach_start = None

    results = []
    resets = []

    print('开始稳定性检测...')
    for ts, v in ratio_series.items():
        ma_fast, _ = ma_fast_calc.update(v)
        ma_slow, std_slow = ma_slow_calc.update(v)
        std_slow_eff = max(std_slow, std_floor)
        d = abs(ma_fast - ma_slow)

        if not state_unstable:
            if d > enter_threshold_mult * std_slow_eff:
                if enter_breach_start is None:
                    enter_breach_start = ts
                duration_ok = (ts - enter_breach_start) >= pd.Timedelta(seconds=enter_confirm_secs)
                if duration_ok:
                    state_unstable = True
                    cooldown_until = ts + wait_delta
                    stable_secs = 0
                    enter_breach_start = None
                    ma_fast_calc.reset(); ma_slow_calc.reset()
                    ma_fast_calc.update(v); ma_slow_calc.update(v)
                    print(
                        f'不稳定开始 {ts}, 超过{enter_threshold_mult}×std已持续≥{enter_confirm_secs}s，阈值= '
                        f'{enter_threshold_mult * std_slow_eff:.6f}，进入等待{wait_minutes}分钟'
                    )
                    resets.append((ts, 'start'))
            else:
                enter_breach_start = None
        else:
            if d > reset_threshold_mult * std_slow_eff:
                cooldown_until = ts + wait_delta
                stable_secs = 0
                ma_fast_calc.reset(); ma_slow_calc.reset()
                ma_fast_calc.update(v); ma_slow_calc.update(v)
                print(f'再次重置: {ts}, 差距={d:.6f} > {reset_threshold_mult}×std，重新等待{wait_minutes}分钟')
                resets.append((ts, 'reset'))
            else:
                if cooldown_until is not None and ts >= cooldown_until:
                    if d < stable_threshold_mult * std_slow_eff:
                        stable_secs += 1
                    else:
                        stable_secs = 0
                    if stable_secs >= stable_need_secs:
                        state_unstable = False
                        print(f'不稳定结束 {ts}, 低于{stable_threshold_mult}×std已持续{stable_need_secs}秒')
                        cooldown_until = None
                        stable_secs = 0

        results.append({
            'timestamp': ts,
            'value': v,
            'ma_fast': ma_fast,
            'ma_slow': ma_slow,
            'std_slow_base': std_slow,
            'ma_diff': d,
            'is_unstable': state_unstable,
            'cooldown_until': cooldown_until,
        })

    print('稳定性检测完成')
    return pd.DataFrame(results), resets

def task(
    symbol: str,
    enter_threshold_mult: float = 0.5,
    reset_threshold_mult: float = 0.3,
    stable_threshold_mult: float = 0.1,
    alpha_fast: float | None = None,
    alpha_slow: float | None = None,
    min_periods: int = 600,
    wait_minutes: int = 30,
    enter_confirm_secs: int = 60,
    std_floor: float = 1e-4,
    stable_need_secs: int = 180,
    binance_dir: str = 'data/binance/',
    bitget_dir: str = 'data/bitget/',
    output_dir: str = './data/images',
):
    # 读取数据
    ratio = get_ratio_data(symbol, binance_dir, bitget_dir)
    if ratio is None:
        print(f"无法处理 {symbol}，数据加载失败")
        return
    
    # 运行检测
    result_df, reset_events = stability_detection_v3(
        ratio,
        alpha_fast=alpha_fast,
        alpha_slow=alpha_slow,
        min_periods=min_periods,
        wait_minutes=wait_minutes,
        enter_confirm_secs=enter_confirm_secs,
        std_floor=std_floor,
        enter_threshold_mult=enter_threshold_mult,
        reset_threshold_mult=reset_threshold_mult,
        stable_threshold_mult=stable_threshold_mult,
        stable_need_secs=stable_need_secs,
    )
    result_df.set_index('timestamp', inplace=True)

    # 绘制结果并保存
    plot_results(symbol, result_df, output_dir=output_dir)

if __name__ == '__main__':
    # 使用多进程
    import multiprocessing
    from multiprocessing import Pool
    import argparse
    
    symbols = ["BNBUSDT", "AAVEUSDT"]

    # 命令行参数：阈值乘数
    parser = argparse.ArgumentParser(description='稳定性检测阈值乘数配置')
    parser.add_argument('--enter-mult', type=float, default=0.5,
                        help='进入不稳定阈值乘数，默认 0.5')
    parser.add_argument('--reset-mult', type=float, default=0.3,
                        help='不稳定阶段重置阈值乘数，默认 0.3')
    parser.add_argument('--stable-mult', type=float, default=0.1,
                        help='恢复稳定判定阈值乘数，默认 0.1')
    # 其他参数
    parser.add_argument('--symbols', nargs='+', default=["BNBUSDT", "AAVEUSDT"], help='交易对列表')
    parser.add_argument('--processes', type=int, default=None, help='并行进程数，默认按CPU和symbols取小')
    parser.add_argument('--alpha-fast', type=float, default=None, help='快速EMA的alpha（默认1小时换算）')
    parser.add_argument('--alpha-slow', type=float, default=None, help='慢速EMA的alpha（默认4小时换算）')
    parser.add_argument('--min-periods', type=int, default=600, help='初始化简单均值/方差的样本数')
    parser.add_argument('--wait-minutes', type=int, default=30, help='进入不稳定后等待分钟数')
    parser.add_argument('--enter-confirm-secs', type=int, default=60, help='进入不稳定阈值需持续秒数')
    parser.add_argument('--std-floor', type=float, default=1e-4, help='标准差下限以避免为0')
    parser.add_argument('--stable-need-secs', type=int, default=180, help='恢复稳定需持续秒数')
    parser.add_argument('--binance-dir', type=str, default='data/binance/', help='Binance数据目录')
    parser.add_argument('--bitget-dir', type=str, default='data/bitget/', help='Bitget数据目录')
    parser.add_argument('--output-dir', type=str, default='./data/images', help='图表输出目录')
    args = parser.parse_args()
    enter_mult = args.enter_mult
    reset_mult = args.reset_mult
    stable_mult = args.stable_mult
    symbols = args.symbols
    procs = args.processes or min(len(symbols), os.cpu_count() or 2)
    alpha_fast = args.alpha_fast
    alpha_slow = args.alpha_slow
    min_periods = args.min_periods
    wait_minutes = args.wait_minutes
    enter_confirm_secs = args.enter_confirm_secs
    std_floor = args.std_floor
    stable_need_secs = args.stable_need_secs
    binance_dir = args.binance_dir
    bitget_dir = args.bitget_dir
    output_dir = args.output_dir

    print(f'运行参数：symbols={symbols}, processes={procs}')
    print(f'阈值乘数：enter={enter_mult}, reset={reset_mult}, stable={stable_mult}')

    with Pool(processes=procs) as pool:
        pool.starmap(
            task,
            [(
                sym,
                enter_mult,
                reset_mult,
                stable_mult,
                alpha_fast,
                alpha_slow,
                min_periods,
                wait_minutes,
                enter_confirm_secs,
                std_floor,
                stable_need_secs,
                binance_dir,
                bitget_dir,
                output_dir,
            ) for sym in symbols]
        )
    print('所有任务完成')
