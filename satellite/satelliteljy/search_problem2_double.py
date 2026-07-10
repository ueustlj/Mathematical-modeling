from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from problem2_model import (
    ConstellationConfig,
    ConstellationEvaluator,
    EvaluationResult,
    SimulationResolution,
)


# ============================================================
# 第二题第（3）小问：95% 时间全区域二重覆盖
#
# 严格判据：
# 某时刻只有当目标区域的每一个网格点都被至少 2 颗卫星覆盖，
# 才记为一个“全区域二重覆盖时刻”。
#
# 最终要求：
# double_full_region_time_ratio >= 0.95
#
# 同时保留前两小问的连续单重覆盖要求：
# single_feasible == True
# ============================================================

OUTPUT_DIR = Path("results_double")
OUTPUT_DIR.mkdir(exist_ok=True)

# 搜索范围。若第一轮完全没有接近 95%，可把上限继续扩大。
M_VALUES = list(range(38, 71, 4))       # 轨道面数
N_VALUES = list(range(44, 73, 4))       # 每轨卫星数
INCLINATION_VALUES = [52.0, 53.0, 54.0]

MIN_TOTAL_SATELLITES = 1700
MAX_TOTAL_SATELLITES = 5000

# 第一阶段每个 M 只抽样若干 Walker 相位因子 F。
PHASE_SAMPLE_COUNT = 7

# 每阶段保留的候选数。
KEEP_AFTER_COARSE = 12
KEEP_AFTER_MEDIUM = 10

# 第一阶段只负责排除明显不合适的方案。
COARSE_RESOLUTION = SimulationResolution(
    grid_step_deg=2.0,
    time_step_s=600.0,
    duration_s=24.0 * 3600.0,
)

# 第二阶段在优秀候选附近细化 M、N、i、F。
MEDIUM_RESOLUTION = SimulationResolution(
    grid_step_deg=1.5,
    time_step_s=300.0,
    duration_s=24.0 * 3600.0,
)

# 第三阶段用于输出当前较可信的候选。
FINE_RESOLUTION = SimulationResolution(
    grid_step_deg=1.0,
    time_step_s=60.0,
    duration_s=24.0 * 3600.0,
)


@dataclass(frozen=True)
class CandidateRecord:
    result: EvaluationResult
    config: ConstellationConfig


def joint_feasible(result: EvaluationResult) -> bool:
    """同时满足连续单重覆盖和 95% 时间全区域二重覆盖。"""
    return bool(result.single_feasible and result.double_95_feasible)


def double_ranking_key(result: EvaluationResult) -> tuple:
    """
    二重覆盖专用排序规则。

    1. 同时满足单重与二重要求的方案排最前；
    2. 可行方案中优先总成本低、卫星少；
    3. 不可行方案中优先提高“全区域二重覆盖时间比例”；
    4. 若该指标相同，再比较二重时空覆盖率和平均覆盖重数。
    """
    feasible = joint_feasible(result)

    if feasible:
        return (
            0,
            result.total_cost_yuan,
            result.total_satellites,
            -result.double_full_region_time_ratio,
            -result.double_space_time_ratio,
        )

    return (
        1,
        -result.double_full_region_time_ratio,
        -result.double_space_time_ratio,
        -result.average_multiplicity,
        0 if result.single_feasible else 1,
        result.total_satellites,
    )


def sampled_phase_factors(planes: int) -> list[int]:
    """在 0~M-1 中均匀抽样 F，并强制包含几个常用值。"""
    sampled = {
        0,
        1 % planes,
        (planes // 4) % planes,
        (planes // 2) % planes,
        (planes - 1) % planes,
    }

    values = np.linspace(0, planes - 1, PHASE_SAMPLE_COUNT)
    sampled.update(int(round(value)) % planes for value in values)
    return sorted(sampled)


def phase_neighbourhood(best_phase: int, planes: int) -> list[int]:
    """对第一阶段较好的 F 做局部细化。"""
    values = {
        (best_phase + delta) % planes
        for delta in range(-3, 4)
    }
    values.update(
        {
            0,
            1 % planes,
            (planes // 4) % planes,
            (planes // 2) % planes,
            (3 * planes // 4) % planes,
            (planes - 1) % planes,
        }
    )
    return sorted(values)


def result_row(result: EvaluationResult) -> dict:
    row = result.to_dict()
    row["joint_feasible"] = joint_feasible(result)
    row["double_shortfall_percent"] = max(
        0.0,
        100.0 * (0.95 - result.double_full_region_time_ratio),
    )
    return row


def save_records(filename: str, records: list[CandidateRecord]) -> None:
    if not records:
        return

    path = OUTPUT_DIR / filename
    rows = [result_row(record.result) for record in records]

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"结果已保存：{path.resolve()}")


def print_one(index: int, total: int, result: EvaluationResult) -> None:
    print(
        f"[{index:4d}/{total:4d}] "
        f"M={result.planes:2d}, "
        f"N={result.satellites_per_plane:2d}, "
        f"i={result.inclination_deg:4.1f}, "
        f"F={result.phase_factor:2d}, "
        f"T={result.total_satellites:4d}, "
        f"double-full={100.0*result.double_full_region_time_ratio:8.4f}%, "
        f"double-ST={100.0*result.double_space_time_ratio:8.4f}%, "
        f"single={result.single_feasible}, "
        f"pass={joint_feasible(result)}"
    )


def stage_one() -> list[ConstellationConfig]:
    print("\n第一阶段：粗网格搜索 M、N、i 和抽样 F")
    evaluator = ConstellationEvaluator(COARSE_RESOLUTION)

    configs: list[ConstellationConfig] = []
    for planes in M_VALUES:
        for satellites_per_plane in N_VALUES:
            total = planes * satellites_per_plane
            if not MIN_TOTAL_SATELLITES <= total <= MAX_TOTAL_SATELLITES:
                continue

            for inclination in INCLINATION_VALUES:
                for phase_factor in sampled_phase_factors(planes):
                    configs.append(
                        ConstellationConfig(
                            planes=planes,
                            satellites_per_plane=satellites_per_plane,
                            inclination_deg=inclination,
                            phase_factor=phase_factor,
                        )
                    )

    records: list[CandidateRecord] = []
    start = time.time()

    for index, config in enumerate(configs, start=1):
        result, _ = evaluator.evaluate(config)
        records.append(CandidateRecord(result=result, config=config))
        print_one(index, len(configs), result)

    records.sort(key=lambda item: double_ranking_key(item.result))
    save_records("stage1_coarse_double.csv", records)

    print(f"第一阶段耗时：{time.time() - start:.1f} s")
    return [record.config for record in records[:KEEP_AFTER_COARSE]]


def stage_two(
    coarse_candidates: list[ConstellationConfig],
) -> list[ConstellationConfig]:
    print("\n第二阶段：中等精度局部细化 M、N、i、F")
    evaluator = ConstellationEvaluator(MEDIUM_RESOLUTION)

    unique_configs: dict[tuple, ConstellationConfig] = {}

    for candidate in coarse_candidates:
        plane_values = sorted(
            {
                value
                for value in (
                    candidate.planes - 2,
                    candidate.planes,
                    candidate.planes + 2,
                )
                if value > 0
            }
        )
        satellite_values = sorted(
            {
                value
                for value in (
                    candidate.satellites_per_plane - 2,
                    candidate.satellites_per_plane,
                    candidate.satellites_per_plane + 2,
                )
                if value > 0
            }
        )
        inclination_values = sorted(
            {
                max(40.0, min(60.0, candidate.inclination_deg + delta))
                for delta in (-1.0, 0.0, 1.0)
            }
        )

        for planes in plane_values:
            # 当 M 改变时，先按相对位置缩放原 F，再做邻域搜索。
            scaled_phase = int(
                round(candidate.phase_factor * planes / candidate.planes)
            ) % planes

            local_phase_values = {
                (scaled_phase + delta) % planes
                for delta in range(-2, 3)
            }
            local_phase_values.update(
                {
                    0,
                    1 % planes,
                    (planes // 2) % planes,
                    (planes - 1) % planes,
                }
            )

            for satellites_per_plane in satellite_values:
                total_satellites = planes * satellites_per_plane
                if not (
                    MIN_TOTAL_SATELLITES
                    <= total_satellites
                    <= MAX_TOTAL_SATELLITES
                ):
                    continue

                for inclination in inclination_values:
                    for phase_factor in sorted(local_phase_values):
                        config = ConstellationConfig(
                            planes=planes,
                            satellites_per_plane=satellites_per_plane,
                            inclination_deg=inclination,
                            phase_factor=phase_factor,
                        )
                        key = (
                            config.planes,
                            config.satellites_per_plane,
                            config.inclination_deg,
                            config.phase_factor,
                        )
                        unique_configs[key] = config

    configs = list(unique_configs.values())
    records: list[CandidateRecord] = []
    start = time.time()

    for index, config in enumerate(configs, start=1):
        result, _ = evaluator.evaluate(config)
        records.append(CandidateRecord(result=result, config=config))
        print_one(index, len(configs), result)

    records.sort(key=lambda item: double_ranking_key(item.result))
    save_records("stage2_medium_double.csv", records)

    print(f"第二阶段耗时：{time.time() - start:.1f} s")
    return [record.config for record in records[:KEEP_AFTER_MEDIUM]]


def stage_three(
    medium_candidates: list[ConstellationConfig],
) -> list[CandidateRecord]:
    print("\n第三阶段：1°、60 s 精细验证")
    evaluator = ConstellationEvaluator(FINE_RESOLUTION)

    records: list[CandidateRecord] = []
    start = time.time()

    for index, config in enumerate(medium_candidates, start=1):
        result, _ = evaluator.evaluate(config)
        records.append(CandidateRecord(result=result, config=config))
        print_one(index, len(medium_candidates), result)

    records.sort(key=lambda item: double_ranking_key(item.result))
    save_records("stage3_fine_double.csv", records)

    print(f"第三阶段耗时：{time.time() - start:.1f} s")
    return records


def print_final(records: list[CandidateRecord]) -> None:
    print("\n" + "=" * 78)
    print("第二题第（3）小问：二重覆盖搜索结果")
    print("=" * 78)

    feasible_records = [
        record for record in records
        if joint_feasible(record.result)
    ]

    if feasible_records:
        best = feasible_records[0].result
        print("当前精度下找到可行方案：")
    else:
        best = records[0].result
        print("当前精度下尚未找到严格可行方案，下面是最接近的方案：")

    print(f"M = {best.planes}")
    print(f"N = {best.satellites_per_plane}")
    print(f"i = {best.inclination_deg:.2f}°")
    print(f"F = {best.phase_factor}")
    print(f"卫星总数 = {best.total_satellites}")
    print(
        "全区域二重覆盖时间比例 = "
        f"{100.0*best.double_full_region_time_ratio:.6f}%"
    )
    print(
        "二重覆盖时空比例（仅作辅助指标） = "
        f"{100.0*best.double_space_time_ratio:.6f}%"
    )
    print(f"连续单重覆盖是否通过 = {best.single_feasible}")
    print(f"95% 时间全区域二重覆盖是否通过 = {best.double_95_feasible}")
    print(f"联合约束是否通过 = {joint_feasible(best)}")
    print(f"总成本 = {best.total_cost_yuan / 1e8:.4f} 亿元")
    print(
        "\n下一步：把该组 M、N、i、F 填入 "
        "validate_problem2_double.py，再提高网格和时间精度。"
    )


def main() -> None:
    coarse_candidates = stage_one()
    medium_candidates = stage_two(coarse_candidates)
    final_records = stage_three(medium_candidates)
    print_final(final_records)


if __name__ == "__main__":
    main()
