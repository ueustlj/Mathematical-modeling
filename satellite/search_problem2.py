from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from problem2_model import (
    ConstellationConfig,
    ConstellationEvaluator,
    SimulationResolution,
    ranking_key,
)


OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 搜索范围：先重点检查“增加轨道面、减少每轨卫星”的平衡方案
# ============================================================
M_VALUES = [36, 38, 40, 42, 44]
N_VALUES = [40, 42, 44, 46, 48, 50]
INCLINATION_VALUES = [52.0, 53.0, 54.0]

MIN_TOTAL_SATELLITES = 1450
MAX_TOTAL_SATELLITES = 2050

# 第一阶段只抽样若干 F，避免组合爆炸。
PHASE_SAMPLE_COUNT = 8

# 每一阶段保留多少个候选进入下一阶段。
KEEP_AFTER_COARSE = 20
KEEP_AFTER_MEDIUM = 8

COARSE_RESOLUTION = SimulationResolution(
    grid_step_deg=2.0,
    time_step_s=600.0,
    duration_s=24.0 * 3600.0,
)

MEDIUM_RESOLUTION = SimulationResolution(
    grid_step_deg=1.0,
    time_step_s=120.0,
    duration_s=24.0 * 3600.0,
)

FINE_RESOLUTION = SimulationResolution(
    grid_step_deg=1.0,
    time_step_s=60.0,
    duration_s=24.0 * 3600.0,
)


def sampled_phase_factors(planes: int) -> list[int]:
    values = np.linspace(
        0,
        planes - 1,
        PHASE_SAMPLE_COUNT,
    )
    return sorted({int(round(value)) for value in values})


def save_results(filename: str, rows: list[dict]) -> None:
    if not rows:
        return

    path = OUTPUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"已保存：{path.resolve()}")


def stage_one() -> list[ConstellationConfig]:
    print("\n第一阶段：粗搜索 M、N、i 和抽样相位 F")
    evaluator = ConstellationEvaluator(COARSE_RESOLUTION)

    results = []
    start = time.time()
    configuration_number = 0

    for M in M_VALUES:
        for N in N_VALUES:
            total = M * N
            if not MIN_TOTAL_SATELLITES <= total <= MAX_TOTAL_SATELLITES:
                continue

            for inclination in INCLINATION_VALUES:
                for phase_factor in sampled_phase_factors(M):
                    config = ConstellationConfig(
                        planes=M,
                        satellites_per_plane=N,
                        inclination_deg=inclination,
                        phase_factor=phase_factor,
                    )
                    result, _ = evaluator.evaluate(config)
                    results.append((result, config))
                    configuration_number += 1

                    print(
                        f"[{configuration_number:4d}] "
                        f"M={M:2d}, N={N:2d}, i={inclination:4.1f}, "
                        f"F={phase_factor:2d}, total={total:4d}, "
                        f"full={100*result.single_full_region_time_ratio:8.4f}%, "
                        f"worst={100*result.worst_single_instantaneous_ratio:8.4f}%, "
                        f"pass={result.single_feasible}"
                    )

    results.sort(key=lambda item: ranking_key(item[0]))
    save_results(
        "stage1_coarse_results.csv",
        [item[0].to_dict() for item in results],
    )

    print(f"第一阶段耗时：{time.time()-start:.1f} s")
    return [item[1] for item in results[:KEEP_AFTER_COARSE]]


def phase_neighbourhood(best_phase: int, planes: int) -> list[int]:
    """对粗搜索得到的相位附近做局部搜索，同时保留几个全局参照值。"""

    values = {
        best_phase % planes,
        (best_phase - 2) % planes,
        (best_phase - 1) % planes,
        (best_phase + 1) % planes,
        (best_phase + 2) % planes,
        0,
        1 % planes,
        (planes // 2) % planes,
        (planes - 1) % planes,
    }
    return sorted(values)


def stage_two(coarse_candidates: list[ConstellationConfig]) -> list[ConstellationConfig]:
    print("\n第二阶段：中等精度局部相位搜索")
    evaluator = ConstellationEvaluator(MEDIUM_RESOLUTION)

    unique_configs: dict[tuple, ConstellationConfig] = {}
    for candidate in coarse_candidates:
        for F in phase_neighbourhood(candidate.phase_factor, candidate.planes):
            config = ConstellationConfig(
                planes=candidate.planes,
                satellites_per_plane=candidate.satellites_per_plane,
                inclination_deg=candidate.inclination_deg,
                phase_factor=F,
            )
            key = (
                config.planes,
                config.satellites_per_plane,
                config.inclination_deg,
                config.phase_factor,
            )
            unique_configs[key] = config

    results = []
    total = len(unique_configs)
    for index, config in enumerate(unique_configs.values(), start=1):
        result, _ = evaluator.evaluate(config)
        results.append((result, config))
        print(
            f"[{index:3d}/{total:3d}] "
            f"M={config.planes:2d}, N={config.satellites_per_plane:2d}, "
            f"i={config.inclination_deg:4.1f}, F={config.phase_factor:2d}, "
            f"full={100*result.single_full_region_time_ratio:8.4f}%, "
            f"gap={result.maximum_single_gap_minutes:5.1f} min, "
            f"pass={result.single_feasible}"
        )

    results.sort(key=lambda item: ranking_key(item[0]))
    save_results(
        "stage2_medium_results.csv",
        [item[0].to_dict() for item in results],
    )
    return [item[1] for item in results[:KEEP_AFTER_MEDIUM]]


def stage_three(medium_candidates: list[ConstellationConfig]) -> list[tuple]:
    print("\n第三阶段：1°、1 min 精细筛选")
    evaluator = ConstellationEvaluator(FINE_RESOLUTION)

    results = []
    for index, config in enumerate(medium_candidates, start=1):
        result, _ = evaluator.evaluate(config, progress=False)
        results.append((result, config))
        print(
            f"[{index:2d}/{len(medium_candidates):2d}] "
            f"M={config.planes:2d}, N={config.satellites_per_plane:2d}, "
            f"i={config.inclination_deg:4.1f}, F={config.phase_factor:2d}, "
            f"total={config.total_satellites:4d}, "
            f"full={100*result.single_full_region_time_ratio:9.5f}%, "
            f"pass={result.single_feasible}"
        )

    results.sort(key=lambda item: ranking_key(item[0]))
    save_results(
        "stage3_fine_results.csv",
        [item[0].to_dict() for item in results],
    )
    return results


def main() -> None:
    coarse_candidates = stage_one()
    medium_candidates = stage_two(coarse_candidates)
    final_results = stage_three(medium_candidates)

    print("\n" + "=" * 72)
    print("当前搜索范围内的最佳候选")
    print("=" * 72)
    for rank, (result, config) in enumerate(final_results[:5], start=1):
        print(
            f"#{rank}: M={config.planes}, N={config.satellites_per_plane}, "
            f"i={config.inclination_deg:.1f}°, F={config.phase_factor}, "
            f"总星数={config.total_satellites}, "
            f"全区域单重时间比例={100*result.single_full_region_time_ratio:.5f}%, "
            f"是否通过={result.single_feasible}"
        )

    feasible_results = [item for item in final_results if item[0].single_feasible]
    if feasible_results:
        feasible_results.sort(key=lambda item: item[0].total_satellites)
        result, config = feasible_results[0]
        print("\n找到通过 1°、1 min、24 h 验证的候选：")
        print(config)
        print(f"总卫星数：{config.total_satellites}")
        print("请把该参数复制到 validate_problem2.py，再提高验证精度。")
    else:
        print("\n当前范围内尚无严格可行方案。")
        print("下一步扩大 M 或 N，或在最佳候选附近细调倾角和相位。")


if __name__ == "__main__":
    main()
