#!/usr/bin/env python3
"""统计 HalfCheetah expert 数据集中每条轨迹的 reward sum 和轨迹长度。

默认读取当前目录下的 `experts/HalfCheetah-v2_d4rl.pkl`。

输出内容：
- 每条轨迹的 reward sum
- 每条轨迹长度
- 数据集平均回报
- 数据集平均轨迹长度
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_dataset(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据集文件: {path}")

    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError("数据集格式错误：期望是一个 dict")

    if "rewards" not in data or "lengths" not in data:
        raise KeyError("数据集中必须包含 'rewards' 和 'lengths' 字段")

    return data


def _to_list(x: Any) -> List[Any]:
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 expert 数据集每条轨迹的 reward sum、长度和平均回报"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("experts/Hopper-v2_d4rl.pkl"),
        help="expert 数据集路径，默认: experts/Hopper-v2_d4rl.pkl",
        # default=Path("supplement/Hopper-v2_d4rl.pkl"),
        # help="expert 数据集路径，默认: supplement/Hopper-v2_d4rl.pkl",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="打印全部轨迹，否则只打印前 20 条",
    )
    args = parser.parse_args()

    data = load_dataset(args.data)
    rewards = _to_list(data["rewards"])
    lengths = _to_list(data["lengths"])

    if len(rewards) != len(lengths):
        raise ValueError(
            f"rewards 数量({len(rewards)})与 lengths 数量({len(lengths)})不一致"
        )

    traj_returns: List[float] = []
    traj_lengths: List[int] = []

    print(f"数据集文件: {args.data}")
    print(f"轨迹条数: {len(rewards)}")
    print("=" * 80)

    limit = len(rewards) if args.show_all else min(20, len(rewards))
    for i, (r, l) in enumerate(zip(rewards, lengths)):
        r_arr = np.asarray(r, dtype=np.float64)
        if r_arr.ndim == 0:
            r_sum = float(r_arr)
        else:
            r_sum = float(np.sum(r_arr))

        traj_len = int(l)
        traj_returns.append(r_sum)
        traj_lengths.append(traj_len)

        if i < limit:
            print(f"轨迹 {i:03d}: reward sum = {r_sum:.6f}, length = {traj_len}")

    if limit < len(rewards):
        print(f"... 仅显示前 {limit} 条轨迹，使用 --show-all 可显示全部")

    print("=" * 80)
    print(f"数据集平均回报: {np.mean(traj_returns):.6f}")
    print(f"数据集回报标准差: {np.std(traj_returns):.6f}")
    print(f"数据集平均轨迹长度: {np.mean(traj_lengths):.2f}")
    print(f"数据集轨迹长度标准差: {np.std(traj_lengths):.2f}")
    print(f"总回报和: {np.sum(traj_returns):.6f}")


if __name__ == "__main__":
    main()
