import csv
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise SystemExit(
        "缺少 scipy。请在 VS Code 终端运行：py -m pip install scipy"
    ) from exc


# ============================================================
# Q4(3) 第一步：基准星座的单星退出鲁棒性诊断
#
# 核心思想：
# 若某时刻某地面点只被一颗卫星覆盖，则该卫星是该时空单元的
# “唯一覆盖卫星”。删除这颗卫星后，该时空单元会失去覆盖。
# 因此无需对 1968 颗卫星逐颗重跑完整仿真，只需一次扫描即可
# 得到所有单星退出情形的覆盖可用率。
# ============================================================


# ------------------------------------------------------------
# 1. 星座与物理参数
# ------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
ORBIT_HEIGHT_KM = 550.0
ORBIT_RADIUS_KM = EARTH_RADIUS_KM + ORBIT_HEIGHT_KM
MU_KM3_S2 = 398600.4418
EARTH_ROTATION_RATE = 7.2921159e-5

M = 41
N = 48
INCLINATION_DEG = 53.0
PHASE_FACTOR = 1
TOTAL_SATELLITES = M * N

COVERAGE_RADIUS_KM = 506.0
COVERAGE_ANGLE_RAD = COVERAGE_RADIUS_KM / EARTH_RADIUS_KM
COVERAGE_CHORD = 2.0 * np.sin(COVERAGE_ANGLE_RAD / 2.0)

MEAN_MOTION = np.sqrt(MU_KM3_S2 / ORBIT_RADIUS_KM**3)


# ------------------------------------------------------------
# 2. 目标区域与仿真精度
# ------------------------------------------------------------
LAT_MIN_DEG = 4.0
LAT_MAX_DEG = 53.0
LON_MIN_DEG = 73.0
LON_MAX_DEG = 135.0

# 第一步先做快速诊断。结果确认后再做精细验证。
GRID_STEP_DEG = 2.0
TIME_STEP_S = 300.0
SIMULATION_DURATION_S = 24.0 * 3600.0

SHOW_FIGURES = True
OUTPUT_DIR = Path("Q4_3_step1_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 3. 基础数组
# ------------------------------------------------------------
PLANE_IDS = np.repeat(np.arange(M), N)
SLOT_IDS = np.tile(np.arange(N), M)

RAAN = 2.0 * np.pi * PLANE_IDS / M
PHASE_IN_PLANE = 2.0 * np.pi * SLOT_IDS / N
PLANE_PHASE = (
    2.0 * np.pi * PHASE_FACTOR * PLANE_IDS / (M * N)
)


# ------------------------------------------------------------
# 4. 网格与位置函数
# ------------------------------------------------------------
def inclusive_axis(v_min, v_max, step):
    values = list(np.arange(v_min, v_max + 1e-12, step))
    if values[-1] < v_max - 1e-9:
        values.append(v_max)
    return np.asarray(values, dtype=float)


def build_ground_grid():
    latitudes = inclusive_axis(LAT_MIN_DEG, LAT_MAX_DEG, GRID_STEP_DEG)
    longitudes = inclusive_axis(LON_MIN_DEG, LON_MAX_DEG, GRID_STEP_DEG)

    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    # 球面面积权重与 cos(latitude) 成正比。
    area_weights = np.cos(np.deg2rad(lat_flat))

    return {
        "latitudes": latitudes,
        "longitudes": longitudes,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "lat_flat": lat_flat,
        "lon_flat": lon_flat,
        "area_weights": area_weights,
    }


def satellite_unit_vectors(t):
    argument = (
        MEAN_MOTION * t
        + PHASE_IN_PLANE
        + PLANE_PHASE
    )

    inc = np.deg2rad(INCLINATION_DEG)

    x_orbit = np.cos(argument)
    y_orbit = np.sin(argument)

    x_inclined = x_orbit
    y_inclined = y_orbit * np.cos(inc)
    z_inclined = y_orbit * np.sin(inc)

    x_eci = x_inclined * np.cos(RAAN) - y_inclined * np.sin(RAAN)
    y_eci = x_inclined * np.sin(RAAN) + y_inclined * np.cos(RAAN)
    z_eci = z_inclined

    vectors = np.column_stack((x_eci, y_eci, z_eci))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def ground_unit_vectors(lat_flat, lon_flat, t):
    lat = np.deg2rad(lat_flat)
    lon = np.deg2rad(lon_flat) + EARTH_ROTATION_RATE * t

    cos_lat = np.cos(lat)

    return np.column_stack((
        cos_lat * np.cos(lon),
        cos_lat * np.sin(lon),
        np.sin(lat),
    ))


def query_neighbours(tree, ground_vectors):
    try:
        return tree.query_ball_point(
            ground_vectors,
            r=COVERAGE_CHORD,
            workers=-1,
        )
    except TypeError:
        return tree.query_ball_point(
            ground_vectors,
            r=COVERAGE_CHORD,
        )


# ------------------------------------------------------------
# 5. 连续区间工具
# ------------------------------------------------------------
def maximum_circular_true_run(boolean_row):
    n = len(boolean_row)

    if not np.any(boolean_row):
        return 0

    if np.all(boolean_row):
        return n

    doubled = np.concatenate((boolean_row, boolean_row))

    best = 0
    current = 0

    for value in doubled:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return min(best, n)


# ------------------------------------------------------------
# 6. 主仿真
# ------------------------------------------------------------
def run_diagnostic():
    grid = build_ground_grid()

    lat_flat = grid["lat_flat"]
    lon_flat = grid["lon_flat"]
    area_weights = grid["area_weights"]

    ground_number = len(lat_flat)
    area_weight_sum = float(np.sum(area_weights))

    times = np.arange(
        0.0,
        SIMULATION_DURATION_S,
        TIME_STEP_S,
    )
    time_number = len(times)

    # 唯一覆盖统计：卫星 × 地面点、卫星 × 时刻。
    critical_count_by_sat_point = np.zeros(
        (TOTAL_SATELLITES, ground_number),
        dtype=np.uint16,
    )

    critical_weight_by_sat_time = np.zeros(
        (TOTAL_SATELLITES, time_number),
        dtype=np.float32,
    )

    critical_cell_count = np.zeros(
        TOTAL_SATELLITES,
        dtype=np.int64,
    )

    baseline_weighted_covered_by_time = np.zeros(
        time_number,
        dtype=float,
    )

    baseline_full_region_by_time = np.zeros(
        time_number,
        dtype=bool,
    )

    baseline_weighted_double_by_time = np.zeros(
        time_number,
        dtype=float,
    )

    baseline_minimum_count = np.iinfo(np.int16).max
    total_unique_cells = 0

    print("=" * 72)
    print("Q4(3) 第一步：单星退出鲁棒性诊断")
    print("=" * 72)
    print(f"星座参数：M={M}, N={N}, i={INCLINATION_DEG:.1f}°, F={PHASE_FACTOR}")
    print(f"卫星总数：{TOTAL_SATELLITES}")
    print(f"网格步长：{GRID_STEP_DEG:.2f}°")
    print(f"时间步长：{TIME_STEP_S:.0f} s")
    print(f"地面点数：{ground_number}")
    print(f"时间样本数：{time_number}")
    print("-" * 72)

    start = time.time()

    for time_id, t in enumerate(times):
        sat_vectors = satellite_unit_vectors(float(t))
        ground_vectors = ground_unit_vectors(
            lat_flat,
            lon_flat,
            float(t),
        )

        tree = cKDTree(sat_vectors)
        neighbour_lists = query_neighbours(tree, ground_vectors)

        counts = np.fromiter(
            (len(items) for items in neighbour_lists),
            dtype=np.int16,
            count=ground_number,
        )

        covered_mask = counts >= 1
        double_mask = counts >= 2

        baseline_weighted_covered_by_time[time_id] = (
            np.sum(area_weights[covered_mask]) / area_weight_sum
        )
        baseline_weighted_double_by_time[time_id] = (
            np.sum(area_weights[double_mask]) / area_weight_sum
        )
        baseline_full_region_by_time[time_id] = bool(np.all(covered_mask))
        baseline_minimum_count = min(
            baseline_minimum_count,
            int(np.min(counts)),
        )

        unique_point_ids = np.flatnonzero(counts == 1)

        if unique_point_ids.size > 0:
            sole_satellites = np.fromiter(
                (
                    neighbour_lists[point_id][0]
                    for point_id in unique_point_ids
                ),
                dtype=np.int32,
                count=unique_point_ids.size,
            )

            unique_weights = area_weights[unique_point_ids]

            np.add.at(
                critical_cell_count,
                sole_satellites,
                1,
            )

            np.add.at(
                critical_count_by_sat_point,
                (sole_satellites, unique_point_ids),
                1,
            )

            np.add.at(
                critical_weight_by_sat_time[:, time_id],
                sole_satellites,
                unique_weights,
            )

            total_unique_cells += int(unique_point_ids.size)

        if (
            time_id % max(1, time_number // 10) == 0
            or time_id == time_number - 1
        ):
            progress = 100.0 * (time_id + 1) / time_number
            elapsed = time.time() - start
            print(
                f"进度：{progress:6.2f}% | "
                f"累计用时：{elapsed:8.2f} s"
            )

    # --------------------------------------------------------
    # 由唯一覆盖统计一次性推导所有单星退出结果
    # --------------------------------------------------------
    baseline_space_time_coverage = float(
        np.mean(baseline_weighted_covered_by_time)
    )
    baseline_full_region_coverage = float(
        np.mean(baseline_full_region_by_time)
    )
    baseline_space_time_double = float(
        np.mean(baseline_weighted_double_by_time)
    )

    critical_weight_total_by_sat = np.sum(
        critical_weight_by_sat_time,
        axis=1,
        dtype=float,
    )

    space_time_after_failure = (
        np.sum(baseline_weighted_covered_by_time)
        - critical_weight_total_by_sat / area_weight_sum
    ) / time_number

    critical_time_mask = critical_weight_by_sat_time > 0.0

    full_region_after_failure = (
        np.sum(baseline_full_region_by_time)
        - np.sum(
            critical_time_mask & baseline_full_region_by_time[None, :],
            axis=1,
        )
    ) / time_number

    point_coverage_after_failure = (
        1.0
        - critical_count_by_sat_point.astype(float) / time_number
    )
    minimum_point_after_failure = np.min(
        point_coverage_after_failure,
        axis=1,
    )

    instantaneous_after_failure = (
        baseline_weighted_covered_by_time[None, :]
        - critical_weight_by_sat_time / area_weight_sum
    )
    worst_instant_after_failure = np.min(
        instantaneous_after_failure,
        axis=1,
    )

    max_outage_steps = np.fromiter(
        (
            maximum_circular_true_run(critical_time_mask[sat_id])
            for sat_id in range(TOTAL_SATELLITES)
        ),
        dtype=np.int32,
        count=TOTAL_SATELLITES,
    )
    max_outage_minutes = max_outage_steps * TIME_STEP_S / 60.0

    pass_full_region_99 = full_region_after_failure >= 0.99 - 1e-12
    pass_min_point_99 = minimum_point_after_failure >= 0.99 - 1e-12
    pass_both_99 = pass_full_region_99 & pass_min_point_99

    worst_full_sat = int(np.argmin(full_region_after_failure))
    worst_point_sat = int(np.argmin(minimum_point_after_failure))
    worst_instant_sat = int(np.argmin(worst_instant_after_failure))

    elapsed_total = time.time() - start

    results = {
        "grid": grid,
        "times": times,
        "baseline_space_time_coverage": baseline_space_time_coverage,
        "baseline_full_region_coverage": baseline_full_region_coverage,
        "baseline_space_time_double": baseline_space_time_double,
        "baseline_minimum_count": baseline_minimum_count,
        "total_unique_cells": total_unique_cells,
        "critical_cell_count": critical_cell_count,
        "space_time_after_failure": space_time_after_failure,
        "full_region_after_failure": full_region_after_failure,
        "minimum_point_after_failure": minimum_point_after_failure,
        "worst_instant_after_failure": worst_instant_after_failure,
        "max_outage_minutes": max_outage_minutes,
        "pass_full_region_99": pass_full_region_99,
        "pass_min_point_99": pass_min_point_99,
        "pass_both_99": pass_both_99,
        "worst_full_sat": worst_full_sat,
        "worst_point_sat": worst_point_sat,
        "worst_instant_sat": worst_instant_sat,
        "elapsed_total": elapsed_total,
    }

    return results


# ------------------------------------------------------------
# 7. 输出文件
# ------------------------------------------------------------
def save_satellite_csv(results):
    path = OUTPUT_DIR / "Q4_3_step1_single_satellite_results.csv"

    fieldnames = [
        "satellite_id",
        "plane_id",
        "slot_id",
        "critical_space_time_cells",
        "space_time_coverage_after_failure",
        "full_region_time_coverage_after_failure",
        "minimum_point_time_coverage_after_failure",
        "worst_instantaneous_area_coverage_after_failure",
        "maximum_full_region_outage_minutes",
        "pass_full_region_99",
        "pass_minimum_point_99",
        "pass_both_99",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for sat_id in range(TOTAL_SATELLITES):
            writer.writerow({
                "satellite_id": sat_id,
                "plane_id": sat_id // N,
                "slot_id": sat_id % N,
                "critical_space_time_cells": int(
                    results["critical_cell_count"][sat_id]
                ),
                "space_time_coverage_after_failure": float(
                    results["space_time_after_failure"][sat_id]
                ),
                "full_region_time_coverage_after_failure": float(
                    results["full_region_after_failure"][sat_id]
                ),
                "minimum_point_time_coverage_after_failure": float(
                    results["minimum_point_after_failure"][sat_id]
                ),
                "worst_instantaneous_area_coverage_after_failure": float(
                    results["worst_instant_after_failure"][sat_id]
                ),
                "maximum_full_region_outage_minutes": float(
                    results["max_outage_minutes"][sat_id]
                ),
                "pass_full_region_99": bool(
                    results["pass_full_region_99"][sat_id]
                ),
                "pass_minimum_point_99": bool(
                    results["pass_min_point_99"][sat_id]
                ),
                "pass_both_99": bool(
                    results["pass_both_99"][sat_id]
                ),
            })

    return path


def sat_label(sat_id):
    return (
        f"ID={sat_id}, plane={sat_id // N}, slot={sat_id % N}"
    )


def save_summary(results):
    path = OUTPUT_DIR / "Q4_3_step1_summary.txt"

    worst_full_sat = results["worst_full_sat"]
    worst_point_sat = results["worst_point_sat"]
    worst_instant_sat = results["worst_instant_sat"]

    all_pass = bool(np.all(results["pass_both_99"]))

    text = f"""Q4(3) 第一步：基准星座单星退出鲁棒性诊断
{'=' * 72}

一、仿真设置
M = {M}
N = {N}
i = {INCLINATION_DEG:.1f} deg
F = {PHASE_FACTOR}
卫星总数 = {TOTAL_SATELLITES}
空间网格步长 = {GRID_STEP_DEG:.2f} deg
时间步长 = {TIME_STEP_S:.0f} s
仿真时长 = {SIMULATION_DURATION_S / 3600.0:.2f} h
地面网格点数 = {results['grid']['lat_flat'].size}
时间样本数 = {results['times'].size}

二、基准星座覆盖结果
面积加权时空单重覆盖率 = {100.0 * results['baseline_space_time_coverage']:.8f} %
全区域同时单重覆盖时间比例 = {100.0 * results['baseline_full_region_coverage']:.8f} %
面积加权时空二重覆盖率 = {100.0 * results['baseline_space_time_double']:.8f} %
全时空最低覆盖重数 = {results['baseline_minimum_count']}
唯一覆盖时空单元总数 = {results['total_unique_cells']}

三、单颗卫星完全退出后的最坏结果
最差全区域时间可用率 = {100.0 * np.min(results['full_region_after_failure']):.8f} %
对应卫星 = {sat_label(worst_full_sat)}

最差地点时间可用率的最小值 = {100.0 * np.min(results['minimum_point_after_failure']):.8f} %
对应卫星 = {sat_label(worst_point_sat)}

最差瞬时面积覆盖率 = {100.0 * np.min(results['worst_instant_after_failure']):.8f} %
对应卫星 = {sat_label(worst_instant_sat)}

最大连续全区域降级时间 = {np.max(results['max_outage_minutes']):.4f} min

通过“全区域时间可用率 >= 99%”的卫星数 = {int(np.sum(results['pass_full_region_99']))} / {TOTAL_SATELLITES}
通过“最差地点时间可用率 >= 99%”的卫星数 = {int(np.sum(results['pass_min_point_99']))} / {TOTAL_SATELLITES}
同时通过两项 99% 约束的卫星数 = {int(np.sum(results['pass_both_99']))} / {TOTAL_SATELLITES}

四、阶段性判定
所有单星退出情形是否均通过 99% 约束 = {all_pass}

说明：本步骤把一颗卫星在完整 24 h 内完全退出视为保守故障情形。
若该保守情形仍通过，则短时避撞造成的 50% 容量下降通常更容易满足覆盖要求；
若未通过，则下一步需要比较 41×49 与 42×48 两种在轨冗余构型。

总计算用时 = {results['elapsed_total']:.2f} s
{'=' * 72}
"""

    path.write_text(text, encoding="utf-8")
    print(text)
    return path


def save_plots(results):
    availability_percent = (
        100.0 * results["full_region_after_failure"]
    )

    plt.figure(figsize=(10, 5.5))
    plt.plot(np.arange(TOTAL_SATELLITES), availability_percent)
    plt.axhline(99.0, linestyle="--", label="99% requirement")
    plt.xlabel("Satellite ID")
    plt.ylabel("Full-region availability after failure (%)")
    plt.title("Single-Satellite Failure Robustness")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "Q4_3_step1_failure_availability.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    outage_grid = (
        100.0 * (1.0 - results["full_region_after_failure"])
    ).reshape(M, N)

    plt.figure(figsize=(11, 6.5))
    image = plt.imshow(outage_grid, aspect="auto", origin="lower")
    plt.colorbar(image, label="Full-region outage ratio (%)")
    plt.xlabel("Slot ID in plane")
    plt.ylabel("Plane ID")
    plt.title("Failure Sensitivity by Plane and Slot")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "Q4_3_step1_plane_slot_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    results = run_diagnostic()
    csv_path = save_satellite_csv(results)
    summary_path = save_summary(results)
    save_plots(results)

    print(f"CSV 已保存：{csv_path}")
    print(f"总结已保存：{summary_path}")
    print(f"图片已保存到：{OUTPUT_DIR}")

    if SHOW_FIGURES:
        image = plt.imread(
            OUTPUT_DIR / "Q4_3_step1_failure_availability.png"
        )
        plt.figure(figsize=(10, 5.5))
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()