import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise SystemExit(
        "缺少 scipy。请在 VS Code 终端运行：python -m pip install scipy"
    ) from exc


# ============================================================
# Q4(3) 第二步：7 天故障窗口的精细鲁棒性验证
#
# 目的：
# 1. 将空间网格由 2° 加密到 1°；
# 2. 将时间步长由 300 s 缩短到 120 s；
# 3. 将仿真时长由 24 h 延长到 7 d；
# 4. 保守地假设某颗卫星在整个 7 天轨道调整期内完全退出。
#
# 若所有单星退出情形仍满足 99% 时间可用率，则说明基准
# 41×48 星座不依赖额外在轨冗余，也能满足题设的 99% 指标。
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
# 2. 目标区域与精细仿真设置
# ------------------------------------------------------------
LAT_MIN_DEG = 4.0
LAT_MAX_DEG = 53.0
LON_MIN_DEG = 73.0
LON_MAX_DEG = 135.0

GRID_STEP_DEG = 1.0
TIME_STEP_S = 120.0
SIMULATION_DAYS = 7.0
SIMULATION_DURATION_S = SIMULATION_DAYS * 24.0 * 3600.0

RELIABILITY_THRESHOLD = 0.99
SHOW_FIGURES = True

OUTPUT_DIR = Path("Q4_3_step2_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 3. 星座基础数组
# ------------------------------------------------------------
PLANE_IDS = np.repeat(np.arange(M), N)
SLOT_IDS = np.tile(np.arange(N), M)

RAAN = 2.0 * np.pi * PLANE_IDS / M
PHASE_IN_PLANE = 2.0 * np.pi * SLOT_IDS / N
PLANE_PHASE = 2.0 * np.pi * PHASE_FACTOR * PLANE_IDS / (M * N)


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

    # 等经纬度网格的面积权重与 cos(latitude) 成正比。
    area_weights = np.cos(np.deg2rad(lat_flat))

    return {
        "latitudes": latitudes,
        "longitudes": longitudes,
        "lat_flat": lat_flat,
        "lon_flat": lon_flat,
        "area_weights": area_weights,
    }


def satellite_unit_vectors(t):
    argument = MEAN_MOTION * t + PHASE_IN_PLANE + PLANE_PHASE
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


def maximum_linear_true_run(boolean_row):
    best = 0
    current = 0

    for value in boolean_row:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


# ------------------------------------------------------------
# 5. 主仿真
# ------------------------------------------------------------
def run_validation():
    grid = build_ground_grid()

    lat_flat = grid["lat_flat"]
    lon_flat = grid["lon_flat"]
    area_weights = grid["area_weights"]

    ground_number = len(lat_flat)
    area_weight_sum = float(np.sum(area_weights))

    times = np.arange(0.0, SIMULATION_DURATION_S, TIME_STEP_S)
    time_number = len(times)

    # 基准星座统计
    baseline_weighted_coverage = np.zeros(time_number, dtype=np.float64)
    baseline_weighted_double = np.zeros(time_number, dtype=np.float64)
    baseline_full_region = np.zeros(time_number, dtype=bool)
    baseline_covered_count_by_point = np.zeros(ground_number, dtype=np.uint32)

    # “唯一覆盖”统计：某颗卫星退出后，只有这些时空单元会失去覆盖
    critical_count_by_sat_point = np.zeros(
        (TOTAL_SATELLITES, ground_number),
        dtype=np.uint16,
    )
    critical_weight_by_sat_time = np.zeros(
        (TOTAL_SATELLITES, time_number),
        dtype=np.float32,
    )
    critical_cell_count = np.zeros(TOTAL_SATELLITES, dtype=np.int64)

    print("=" * 76)
    print("Q4(3) 第二步：7 天故障窗口精细鲁棒性验证")
    print("=" * 76)
    print(
        f"星座参数：M={M}, N={N}, i={INCLINATION_DEG:.1f}°, "
        f"F={PHASE_FACTOR}"
    )
    print(f"卫星总数：{TOTAL_SATELLITES}")
    print(f"空间网格步长：{GRID_STEP_DEG:.2f}°")
    print(f"时间步长：{TIME_STEP_S:.0f} s")
    print(f"仿真时长：{SIMULATION_DAYS:.1f} d")
    print(f"地面网格点数：{ground_number}")
    print(f"时间样本数：{time_number}")
    print("-" * 76)

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

        baseline_covered_count_by_point += covered_mask.astype(np.uint32)

        baseline_weighted_coverage[time_id] = (
            np.sum(area_weights[covered_mask]) / area_weight_sum
        )
        baseline_weighted_double[time_id] = (
            np.sum(area_weights[double_mask]) / area_weight_sum
        )
        baseline_full_region[time_id] = bool(np.all(covered_mask))

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

            np.add.at(critical_cell_count, sole_satellites, 1)
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

        if (
            time_id % max(1, time_number // 20) == 0
            or time_id == time_number - 1
        ):
            progress = 100.0 * (time_id + 1) / time_number
            elapsed = time.time() - start
            print(
                f"进度：{progress:6.2f}% | "
                f"累计用时：{elapsed:8.2f} s"
            )

    # --------------------------------------------------------
    # 6. 推导每颗卫星退出后的指标
    # --------------------------------------------------------
    baseline_space_time_coverage = float(
        np.mean(baseline_weighted_coverage)
    )
    baseline_full_region_availability = float(
        np.mean(baseline_full_region)
    )
    baseline_space_time_double = float(
        np.mean(baseline_weighted_double)
    )

    critical_weight_total_by_sat = np.sum(
        critical_weight_by_sat_time,
        axis=1,
        dtype=np.float64,
    )

    space_time_after_failure = (
        np.sum(baseline_weighted_coverage)
        - critical_weight_total_by_sat / area_weight_sum
    ) / time_number

    critical_time_mask = critical_weight_by_sat_time > 0.0

    # 基准星座全区域有覆盖，且该时刻不存在该卫星的唯一覆盖单元时，
    # 故障后全区域仍有覆盖。
    full_region_state_after_failure = (
        baseline_full_region[None, :]
        & (~critical_time_mask)
    )
    full_region_after_failure = np.mean(
        full_region_state_after_failure,
        axis=1,
    )

    point_coverage_after_failure = (
        baseline_covered_count_by_point[None, :].astype(np.float64)
        - critical_count_by_sat_point.astype(np.float64)
    ) / time_number
    minimum_point_after_failure = np.min(
        point_coverage_after_failure,
        axis=1,
    )

    instantaneous_after_failure = (
        baseline_weighted_coverage[None, :]
        - critical_weight_by_sat_time / area_weight_sum
    )
    worst_instant_after_failure = np.min(
        instantaneous_after_failure,
        axis=1,
    )

    max_outage_steps = np.fromiter(
        (
            maximum_linear_true_run(critical_time_mask[sat_id])
            for sat_id in range(TOTAL_SATELLITES)
        ),
        dtype=np.int32,
        count=TOTAL_SATELLITES,
    )
    max_outage_minutes = max_outage_steps * TIME_STEP_S / 60.0

    pass_full_region_99 = (
        full_region_after_failure >= RELIABILITY_THRESHOLD - 1e-12
    )
    pass_min_point_99 = (
        minimum_point_after_failure >= RELIABILITY_THRESHOLD - 1e-12
    )
    pass_both_99 = pass_full_region_99 & pass_min_point_99

    worst_full_sat = int(np.argmin(full_region_after_failure))
    worst_point_sat = int(np.argmin(minimum_point_after_failure))
    worst_instant_sat = int(np.argmin(worst_instant_after_failure))

    # 最差卫星逐日全区域时间可用率
    worst_state = full_region_state_after_failure[worst_full_sat]
    samples_per_day = int(round(24.0 * 3600.0 / TIME_STEP_S))
    daily_availability = []

    for day_id in range(int(SIMULATION_DAYS)):
        left = day_id * samples_per_day
        right = min((day_id + 1) * samples_per_day, time_number)
        daily_availability.append(float(np.mean(worst_state[left:right])))

    elapsed_total = time.time() - start

    return {
        "grid": grid,
        "times": times,
        "baseline_space_time_coverage": baseline_space_time_coverage,
        "baseline_full_region_availability": baseline_full_region_availability,
        "baseline_space_time_double": baseline_space_time_double,
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
        "daily_availability": daily_availability,
        "elapsed_total": elapsed_total,
    }


# ------------------------------------------------------------
# 7. 文件输出
# ------------------------------------------------------------
def sat_label(sat_id):
    return f"ID={sat_id}, plane={sat_id // N}, slot={sat_id % N}"


def save_csv(results):
    path = OUTPUT_DIR / "Q4_3_step2_single_satellite_results.csv"

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


def save_summary(results):
    path = OUTPUT_DIR / "Q4_3_step2_summary.txt"

    worst_full_sat = results["worst_full_sat"]
    worst_point_sat = results["worst_point_sat"]
    worst_instant_sat = results["worst_instant_sat"]

    all_pass = bool(np.all(results["pass_both_99"]))
    critical_satellite_number = int(
        np.sum(results["critical_cell_count"] > 0)
    )

    daily_text = "\n".join(
        (
            f"第 {day_id + 1} 天：{availability * 100:.8f} %"
            for day_id, availability in enumerate(
                results["daily_availability"]
            )
        )
    )

    text = f"""Q4(3) 第二步：7 天故障窗口精细鲁棒性验证
{'=' * 76}

一、仿真设置
M = {M}
N = {N}
i = {INCLINATION_DEG:.1f} deg
F = {PHASE_FACTOR}
卫星总数 = {TOTAL_SATELLITES}
空间网格步长 = {GRID_STEP_DEG:.2f} deg
时间步长 = {TIME_STEP_S:.0f} s
仿真时长 = {SIMULATION_DAYS:.1f} d
可靠性阈值 = {RELIABILITY_THRESHOLD * 100:.2f} %

二、基准星座覆盖结果
面积加权时空单重覆盖率 = {results['baseline_space_time_coverage'] * 100:.8f} %
全区域同时单重覆盖时间比例 = {results['baseline_full_region_availability'] * 100:.8f} %
面积加权时空二重覆盖率 = {results['baseline_space_time_double'] * 100:.8f} %
出现过唯一覆盖的卫星数 = {critical_satellite_number} / {TOTAL_SATELLITES}

三、单颗卫星在完整 7 天内退出后的最坏结果
最差全区域时间可用率 = {results['full_region_after_failure'][worst_full_sat] * 100:.8f} %
对应卫星 = {sat_label(worst_full_sat)}

最差地点时间可用率的最小值 = {results['minimum_point_after_failure'][worst_point_sat] * 100:.8f} %
对应卫星 = {sat_label(worst_point_sat)}

最差瞬时面积覆盖率 = {results['worst_instant_after_failure'][worst_instant_sat] * 100:.8f} %
对应卫星 = {sat_label(worst_instant_sat)}

最大连续全区域降级时间 = {np.max(results['max_outage_minutes']):.4f} min

通过“全区域时间可用率 >= 99%”的卫星数 = {int(np.sum(results['pass_full_region_99']))} / {TOTAL_SATELLITES}
通过“最差地点时间可用率 >= 99%”的卫星数 = {int(np.sum(results['pass_min_point_99']))} / {TOTAL_SATELLITES}
同时通过两项 99% 约束的卫星数 = {int(np.sum(results['pass_both_99']))} / {TOTAL_SATELLITES}

四、最差卫星逐日全区域时间可用率
{daily_text}

五、阶段性判定
所有单星退出情形是否均通过 99% 约束 = {all_pass}

解释：
本步骤将某颗卫星在完整 7 天轨道调整期内完全退出，视为保守故障情形。
若 all_pass=True，则在当前离散精度下，基准 41×48 星座已经满足
“单颗卫星退出后，99% 时间保持基本覆盖”的要求，最小额外在轨冗余数可取 0。
后续仍需比较地面备用星与在轨冗余方案的成本、恢复速度和工程风险。

总计算用时 = {results['elapsed_total']:.2f} s
{'=' * 76}
"""

    path.write_text(text, encoding="utf-8")
    return path


def save_figures(results):
    satellite_ids = np.arange(TOTAL_SATELLITES)

    # 图 1：所有单星退出情形的全区域时间可用率
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        satellite_ids,
        results["full_region_after_failure"] * 100.0,
        linewidth=1.2,
    )
    ax.axhline(
        RELIABILITY_THRESHOLD * 100.0,
        linestyle="--",
        linewidth=1.5,
        label="99% requirement",
    )
    ax.set_xlabel("Satellite ID")
    ax.set_ylabel("Full-region availability after failure (%)")
    ax.set_title("Seven-Day Single-Satellite Failure Robustness")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "Q4_3_step2_failure_availability.png",
        dpi=220,
    )

    # 图 2：轨道面—槽位故障敏感性
    outage_ratio = (
        1.0 - results["full_region_after_failure"]
    ).reshape(M, N) * 100.0

    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(
        outage_ratio,
        origin="lower",
        aspect="auto",
    )
    ax.set_xlabel("Slot ID in plane")
    ax.set_ylabel("Plane ID")
    ax.set_title("Seven-Day Failure Sensitivity by Plane and Slot")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Full-region outage ratio (%)")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "Q4_3_step2_plane_slot_heatmap.png",
        dpi=220,
    )

    # 图 3：最差卫星逐日可用率
    days = np.arange(1, len(results["daily_availability"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        days,
        np.asarray(results["daily_availability"]) * 100.0,
    )
    ax.axhline(
        RELIABILITY_THRESHOLD * 100.0,
        linestyle="--",
        linewidth=1.5,
        label="99% requirement",
    )
    ax.set_xlabel("Day")
    ax.set_ylabel("Full-region availability (%)")
    ax.set_title(
        "Daily Availability for the Worst Failed Satellite"
    )
    ax.set_xticks(days)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "Q4_3_step2_worst_satellite_daily.png",
        dpi=220,
    )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


# ------------------------------------------------------------
# 8. 主程序
# ------------------------------------------------------------
def main():
    results = run_validation()

    csv_path = save_csv(results)
    summary_path = save_summary(results)
    save_figures(results)

    worst_sat = results["worst_full_sat"]
    all_pass = bool(np.all(results["pass_both_99"]))

    print("\n" + "=" * 76)
    print("精细验证完成")
    print("=" * 76)
    print(
        "最差全区域时间可用率 = "
        f"{results['full_region_after_failure'][worst_sat] * 100:.8f} %"
    )
    print(f"对应卫星 = {sat_label(worst_sat)}")
    print(
        "最大连续全区域降级时间 = "
        f"{np.max(results['max_outage_minutes']):.4f} min"
    )
    print(
        f"同时通过两项 99% 约束的卫星数 = "
        f"{int(np.sum(results['pass_both_99']))} / "
        f"{TOTAL_SATELLITES}"
    )
    print(f"所有单星退出情形均通过 = {all_pass}")
    print(f"\nCSV 已保存：{csv_path}")
    print(f"总结已保存：{summary_path}")
    print(f"图片已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()