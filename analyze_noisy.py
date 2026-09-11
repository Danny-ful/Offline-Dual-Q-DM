import numpy as np

# 手动分析eval数据
eval_data_1 = """946.3371111,0
-30.984768600000002,5000
-19.817906200000003,10000
-437.24397369999997,15000
-357.6573677,20000
-1003.7392920000002,25000
-564.5783639,30000
-524.8205555000001,35000
-591.7136927000001,40000
-281.5062796,45000
-93.0595924,50000
-1713.3422890999998,55000
-584.7268277000001,60000
-880.5872474000001,65000
-3001.5061062,70000"""

lines = eval_data_1.strip().split('\n')
rewards = [float(line.split(',')[0]) for line in lines]
steps = [float(line.split(',')[1]) for line in lines]

print("=" * 80)
print("NOISY EXPERT ANT - 性能分析")
print("=" * 80)

print("\n1. 性能轨迹:")
for i, (r, s) in enumerate(zip(rewards, steps)):
    print(f"  Step {s:>7.0f}: {r:>10.2f}")

print(f"\n2. 关键指标:")
print(f"  初始性能 (随机策略): {rewards[0]:.2f}")
print(f"  最高性能: {max(rewards):.2f}")
print(f"  最低性能: {min(rewards):.2f}")
print(f"  早期学习 (5k-65k步) 平均: {np.mean(rewards[1:13]):.2f}")
print(f"  性能崩溃点: ~65000步 (性能跌至 -3001)")
print(f"  崩溃幅度: {rewards[14] - rewards[12]:.2f}")

print(f"\n3. 问题分析:")
print(f"  ✗ 早期学习不稳定: 性能在5k-60k步间大幅波动")
print(f"  ✗ 无法正向改进: 最高只能到 {rewards[1]:.2f}")
print(f"  ✗ 完全性能崩溃: 65k步后性能固定在 -3001 (可能是状态分布外)")
print(f"  ✗ 无法恢复: 后续训练无法恢复性能")

