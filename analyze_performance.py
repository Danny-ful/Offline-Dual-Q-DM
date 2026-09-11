# 手动分析noisy_expert实验的性能问题

eval_data = {
    "exp1_pid449": [
        (946.3371111, 0),
        (-30.984768600000002, 5000),
        (-19.817906200000003, 10000),
        (-437.24397369999997, 15000),
        (-357.6573677, 20000),
        (-1003.7392920000002, 25000),
        (-564.5783639, 30000),
        (-524.8205555000001, 35000),
        (-591.7136927000001, 40000),
        (-281.5062796, 45000),
        (-93.0595924, 50000),
        (-1713.3422890999998, 55000),
        (-584.7268277000001, 60000),
        (-880.5872474000001, 65000),
        (-3001.5061062, 70000),
    ],
    "exp3_pid636": [
        (946.2172277000002, 0),
        (85.2469682, 5000),
        (-10.5479406, 10000),
        (-81.3286766, 15000),
        (-170.5717295, 20000),
        (-155.08577480000002, 25000),
        (-121.93992250000001, 30000),
        (-281.0564543, 35000),
        (-106.951894, 40000),
        (-828.1200693000001, 45000),
        (-504.44011309999996, 50000),
        (-538.4853177, 55000),
        (-55.4509282, 60000),
        (-3001.2952720999997, 65000),
        (-3001.5110193, 70000),
    ]
}

print("=" * 90)
print("NOISY EXPERT ANT 实验性能分析 - 问题诊断")
print("=" * 90)

for exp_name, data in eval_data.items():
    rewards = [r for r, s in data]
    steps = [s for r, s in data]
    
    print(f"\n【{exp_name}】")
    print("-" * 90)
    
    # 1. 基础指标
    print(f"初始性能 (随机策略): {rewards[0]:>10.2f}")
    print(f"最高性能:          {max(rewards):>10.2f}")
    print(f"最低性能:          {min(rewards):>10.2f}")
    print(f"平均性能:          {sum(rewards)/len(rewards):>10.2f}")
    
    # 2. 性能轨迹分析
    print(f"\n性能轨迹 (step -> reward):")
    for i, (r, s) in enumerate(data):
        if i == 0:
            print(f"  {s:>7.0f}步: {r:>10.2f}  ← 随机策略初始值")
        else:
            change = r - rewards[i-1]
            direction = "↓" if change < -100 else ("↑" if change > 50 else "~")
            print(f"  {s:>7.0f}步: {r:>10.2f}  {direction} (Δ={change:>8.2f})")
    
    # 3. 关键问题
    print(f"\n【关键问题】")
    
    # 性能崩溃
    collapse_idx = None
    for i, r in enumerate(rewards):
        if r < -3000:
            collapse_idx = i
            break
    
    if collapse_idx:
        print(f"  ✗ 性能崩溃点: step={steps[collapse_idx]:.0f}")
        print(f"    - 崩溃前性能: {rewards[collapse_idx-1]:.2f}")
        print(f"    - 崩溃后性能: {rewards[collapse_idx]:.2f}")
        print(f"    - 崩溃幅度: {rewards[collapse_idx] - rewards[collapse_idx-1]:.2f}")
    
    # 早期学习能力
    early_rewards = rewards[1:7]  # 5k-30k步
    print(f"  ✗ 早期学习失败: 平均性能 {sum(early_rewards)/len(early_rewards):.2f}")
    print(f"    - 5k步: {rewards[1]:.2f} (应该有正向改进)")
    print(f"    - 30k步: {rewards[6]:.2f} (仍为负值)")
    
    # 稳定性
    pre_collapse = rewards[:collapse_idx] if collapse_idx else rewards
    variance = sum((r - sum(pre_collapse)/len(pre_collapse))**2 for r in pre_collapse) / len(pre_collapse)
    print(f"  ✗ 训练不稳定: 方差={variance:.0f}")
    print(f"    - 性能在 {min(pre_collapse):.2f} 到 {max(pre_collapse):.2f} 之间大幅波动")

print("\n" + "=" * 90)
print("性能差的根本原因总结")
print("=" * 90)
print("""
1【数据质量问题】
   - 50% 随机动作轨迹混入专家数据中
   - IQ-Learn 无法有效区分高质量与低质量演示
   - 导致学到的奖励函数包含噪声

2【学习目标冲突】
   - 模型试图同时学习两个不兼容的策略
   - 梯度信号被随机演示干扰
   - 最终导致策略完全崩溃

3【超参数不适配】
   - 当前超参数是为高质量专家数据调优的
   - 没有针对噪声数据的鲁棒性设计
   - 约束机制无法有效处理异常值

4【状态分布偏移】
   - 随机动作产生的状态分布与专家分布不同
   - 策略在未见过的状态上产生极端负奖励(-3001)
   - 表明行为策略覆盖不足
""")

print("=" * 90)
print("改进方向")
print("=" * 90)
print("""
【立即可做】
  1. 增加 IQ-Learn 的约束强度
     - 提高 constrain=True 的 alpha 参数
     - 或使用 CQL 风格的保守约束

  2. 降低对噪声数据的依赖
     - 减少随机轨迹的权重或采样概率
     - 使用不确定性估计来downweight低质量轨迹

  3. 调整网络容量
     - 减小隐藏层大小 (256 -> 128)
     - 降低学习率以防止过拟合噪声

【需要修改代码】
  4. 实现鲁棒性损失函数
     - Huber loss 替代 MSE
     - Outlier rejection 机制

  5. 添加策略熵正则化
     - 防止策略过度确信于噪声信号
     - 维持探索能力

  6. 分离数据流处理
     - 专家轨迹和补充轨迹分开采样
     - 可视化学习曲线分解
""")

