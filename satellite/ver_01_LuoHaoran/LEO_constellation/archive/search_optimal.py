import csv
import time

import numpy as np

from archive.parameter import (
    a,
    mu,
    omega_earth,
    coverage_angle,
    walker_phase_factor
)


# ============================================================
# 1. 粗搜索参数
# ============================================================

# 搜索的轨道面数量
M_VALUES = [
    8, 12, 16, 20, 24, 28, 32
]

# 搜索的每轨卫星数量
N_VALUES = [
    16, 24, 32, 40, 48, 56, 64
]

# 搜索的轨道倾角
INCLINATION_VALUES = [
    50.0, 53.0, 56.0, 60.0
]


# 粗搜索时降低仿真精度，加快计算
LATITUDE_MIN = 4.0
LATITUDE_MAX = 53.0

LONGITUDE_MIN = 73.0
LONGITUDE_MAX = 135.0

# 经纬度每 6° 取一个网格点
GRID_STEP = 6.0

# 每 15 分钟计算一次
TIME_STEP = 900.0

# 模拟 24 小时
SIMULATION_DURATION = 24.0 * 3600.0


# ============================================================
# 2. 构造目标区域地面网格
# ============================================================

def build_ground_grid():
    """
    构造目标区域网格，并预先计算各时刻地面点
    在地心惯性坐标系中的单位方向向量。
    """

    latitudes = np.arange(
        LATITUDE_MIN,
        LATITUDE_MAX + GRID_STEP,
        GRID_STEP
    )

    longitudes = np.arange(
        LONGITUDE_MIN,
        LONGITUDE_MAX + GRID_STEP,
        GRID_STEP
    )

    longitude_grid, latitude_grid = np.meshgrid(
        longitudes,
        latitudes
    )

    latitude_rad = np.deg2rad(
        latitude_grid.ravel()
    )

    longitude_rad = np.deg2rad(
        longitude_grid.ravel()
    )

    times = np.arange(
        0.0,
        SIMULATION_DURATION,
        TIME_STEP
    )

    ground_unit_vectors = []

    for t in times:

        # 考虑地球自转
        longitude_eci = (
            longitude_rad
            + omega_earth * t
        )

        x = (
            np.cos(latitude_rad)
            * np.cos(longitude_eci)
        )

        y = (
            np.cos(latitude_rad)
            * np.sin(longitude_eci)
        )

        z = np.sin(latitude_rad)

        ground_unit = np.column_stack(
            (x, y, z)
        )

        ground_unit_vectors.append(
            ground_unit
        )

    return (
        times,
        np.array(ground_unit_vectors),
        latitude_grid,
        longitude_grid
    )


# ============================================================
# 3. 快速计算整个星座的位置
# ============================================================

def constellation_positions_fast(
        t,
        plane_number,
        satellites_per_plane,
        inclination_degree
):
    """
    向量化计算某一时刻全部卫星的位置。

    返回形状：
    (卫星总数, 3)
    """

    plane_ids = np.repeat(
        np.arange(plane_number),
        satellites_per_plane
    )

    satellite_ids = np.tile(
        np.arange(satellites_per_plane),
        plane_number
    )

    # 升交点赤经
    raan = (
        2.0
        * np.pi
        * plane_ids
        / plane_number
    )

    # 同一轨道面内的均匀相位
    phase_in_plane = (
        2.0
        * np.pi
        * satellite_ids
        / satellites_per_plane
    )

    # Walker 星座的轨道面间相位差
    plane_phase = (
        2.0
        * np.pi
        * walker_phase_factor
        * plane_ids
        / (
            plane_number
            * satellites_per_plane
        )
    )

    mean_motion = np.sqrt(
        mu / a ** 3
    )

    argument = (
        mean_motion * t
        + phase_in_plane
        + plane_phase
    )

    inclination_rad = np.deg2rad(
        inclination_degree
    )

    # 轨道平面坐标
    x_orbit = a * np.cos(argument)
    y_orbit = a * np.sin(argument)

    # 先绕 x 轴旋转轨道倾角
    x_inclined = x_orbit

    y_inclined = (
        y_orbit
        * np.cos(inclination_rad)
    )

    z_inclined = (
        y_orbit
        * np.sin(inclination_rad)
    )

    # 再绕 z 轴旋转升交点赤经
    x_eci = (
        x_inclined * np.cos(raan)
        - y_inclined * np.sin(raan)
    )

    y_eci = (
        x_inclined * np.sin(raan)
        + y_inclined * np.cos(raan)
    )

    z_eci = z_inclined

    return np.column_stack(
        (x_eci, y_eci, z_eci)
    )


# ============================================================
# 4. 计算周期性最大覆盖间隙
# ============================================================

def calculate_circular_maximum_gap(
        coverage_history
):
    """
    coverage_history 形状：
    (时间步数, 地面点数)

    因为仿真以 24 小时为周期，所以还要处理
    第一天末尾和第二天开头连接产生的覆盖间隙。
    """

    time_number, ground_number = (
        coverage_history.shape
    )

    global_maximum_gap_steps = 0

    for ground_id in range(ground_number):

        covered = coverage_history[
            :,
            ground_id
        ]

        # 始终被覆盖
        if np.all(covered):
            point_maximum_gap = 0

        # 始终没有覆盖
        elif np.all(~covered):
            point_maximum_gap = time_number

        else:
            doubled = np.concatenate(
                (covered, covered)
            )

            current_gap = 0
            point_maximum_gap = 0

            for state in doubled:

                if not state:
                    current_gap += 1

                    point_maximum_gap = max(
                        point_maximum_gap,
                        current_gap
                    )

                else:
                    current_gap = 0

            # 最大值不能超过一个完整仿真周期
            point_maximum_gap = min(
                point_maximum_gap,
                time_number
            )

        global_maximum_gap_steps = max(
            global_maximum_gap_steps,
            point_maximum_gap
        )

    return (
        global_maximum_gap_steps
        * TIME_STEP
        / 60.0
    )


# ============================================================
# 5. 评价一个星座方案
# ============================================================

def evaluate_configuration(
        plane_number,
        satellites_per_plane,
        inclination_degree,
        times,
        ground_unit_vectors
):
    """
    计算一个 M-N-i 组合的覆盖性能。
    """

    time_number = len(times)

    ground_number = (
        ground_unit_vectors.shape[1]
    )

    coverage_history = np.zeros(
        (time_number, ground_number),
        dtype=bool
    )

    coverage_multiplicity_sum = 0

    cosine_limit = np.cos(
        coverage_angle
    )

    for time_id, t in enumerate(times):

        satellite_positions = (
            constellation_positions_fast(
                t=t,
                plane_number=plane_number,
                satellites_per_plane=satellites_per_plane,
                inclination_degree=inclination_degree
            )
        )

        # 卫星轨道半径恒为 a
        satellite_unit_vectors = (
            satellite_positions / a
        )

        cosine_matrix = (
            ground_unit_vectors[time_id]
            @ satellite_unit_vectors.T
        )

        visible_matrix = (
            cosine_matrix >= cosine_limit
        )

        coverage_counts = np.sum(
            visible_matrix,
            axis=1
        )

        coverage_history[time_id] = (
            coverage_counts >= 1
        )

        coverage_multiplicity_sum += np.sum(
            coverage_counts
        )

    # 所有“时间—空间”单元的覆盖率
    space_time_coverage_ratio = np.mean(
        coverage_history
    )

    # 每个时刻整个区域是否全部被覆盖
    full_region_state = np.all(
        coverage_history,
        axis=1
    )

    full_region_coverage_ratio = np.mean(
        full_region_state
    )

    # 每一个时刻被覆盖的区域比例
    instantaneous_coverage_ratio = np.mean(
        coverage_history,
        axis=1
    )

    # 最差时刻的区域覆盖率
    worst_instantaneous_coverage_ratio = np.min(
        instantaneous_coverage_ratio
    )

    # 每个地点在24小时内的覆盖率
    point_time_coverage_ratio = np.mean(
        coverage_history,
        axis=0
    )

    # 最差地点的时间覆盖率
    minimum_point_coverage_ratio = np.min(
        point_time_coverage_ratio
    )

    average_multiplicity = (
        coverage_multiplicity_sum
        / (
            time_number
            * ground_number
        )
    )

    maximum_gap_minutes = (
        calculate_circular_maximum_gap(
            coverage_history
        )
    )

    feasible = (
        full_region_coverage_ratio
        >= 1.0 - 1e-12
        and maximum_gap_minutes
        <= 1e-12
    )

    return {
        "M": plane_number,

        "N": satellites_per_plane,

        "inclination":
            inclination_degree,

        "total_satellites":
            plane_number
            * satellites_per_plane,

        "space_time_coverage_ratio":
            space_time_coverage_ratio,

        "full_region_coverage_ratio":
            full_region_coverage_ratio,

        "worst_instantaneous_coverage_ratio":
            worst_instantaneous_coverage_ratio,

        "minimum_point_coverage_ratio":
            minimum_point_coverage_ratio,

        "average_multiplicity":
            average_multiplicity,

        "maximum_gap_minutes":
            maximum_gap_minutes,

        "feasible":
            feasible
    }


# ============================================================
# 6. 结果排序规则
# ============================================================

def ranking_key(result):
    """
    可行方案优先按照总卫星数从少到多排列。

    不可行方案优先选择：
    1. 最差时刻覆盖率较高；
    2. 最差地点覆盖率较高；
    3. 总体时空覆盖率较高；
    4. 最大覆盖间隙较小。
    """

    if result["feasible"]:

        return (
            0,
            result["total_satellites"],
            -result[
                "average_multiplicity"
            ]
        )

    return (
        1,

        -result[
            "worst_instantaneous_coverage_ratio"
        ],

        -result[
            "minimum_point_coverage_ratio"
        ],

        -result[
            "space_time_coverage_ratio"
        ],

        result[
            "maximum_gap_minutes"
        ],

        result[
            "total_satellites"
        ]
    )


# ============================================================
# 7. 保存搜索结果
# ============================================================

def save_results(results):
    """
    将搜索结果保存为 CSV 文件。
    """

    filename = "coarse_search_results.csv"

    fieldnames = [
        "M",
        "N",
        "inclination",
        "total_satellites",
        "space_time_coverage_ratio",
        "full_region_coverage_ratio",
        "worst_instantaneous_coverage_ratio",
        "minimum_point_coverage_ratio",
        "average_multiplicity",
        "maximum_gap_minutes",
        "feasible"
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    print(
        f"\n全部结果已保存到：{filename}"
    )


# ============================================================
# 8. 打印最优候选方案
# ============================================================

def print_top_results(
        sorted_results,
        number=15
):
    """
    打印排名靠前的候选星座。
    """

    print("\n")
    print("=" * 110)

    print(
        "排名  M   N   倾角   总星数   "
        "时空覆盖率   最差时刻覆盖率   "
        "最差地点覆盖率   最大间隙/min   可行"
    )

    print("=" * 110)

    for rank, result in enumerate(
            sorted_results[:number],
            start=1
    ):

        print(
            f"{rank:>4} "
            f"{result['M']:>3} "
            f"{result['N']:>3} "
            f"{result['inclination']:>6.1f} "
            f"{result['total_satellites']:>8} "
            f"{100 * result['space_time_coverage_ratio']:>11.3f}% "
            f"{100 * result['worst_instantaneous_coverage_ratio']:>15.3f}% "
            f"{100 * result['minimum_point_coverage_ratio']:>15.3f}% "
            f"{result['maximum_gap_minutes']:>13.2f} "
            f"{str(result['feasible']):>7}"
        )

    print("=" * 110)


# ============================================================
# 9. 主搜索程序
# ============================================================

def main():
    """
    搜索满足连续覆盖要求且卫星总数尽可能少的方案。
    """

    print("正在建立地面网格……")

    (
        times,
        ground_unit_vectors,
        latitude_grid,
        longitude_grid
    ) = build_ground_grid()

    configurations = []

    for plane_number in M_VALUES:

        for satellites_per_plane in N_VALUES:

            for inclination_degree in (
                    INCLINATION_VALUES
            ):

                configurations.append(
                    (
                        plane_number,
                        satellites_per_plane,
                        inclination_degree
                    )
                )

    # 按总卫星数从少到多搜索
    configurations.sort(
        key=lambda item: item[0] * item[1]
    )

    total_configuration_number = len(
        configurations
    )

    print(
        f"一共需要测试 "
        f"{total_configuration_number} "
        f"组粗搜索方案。"
    )

    print(
        f"粗搜索网格点数："
        f"{latitude_grid.size}"
    )

    print(
        f"粗搜索时间步数："
        f"{len(times)}"
    )

    print("\n开始搜索……\n")

    start_time = time.time()

    results = []

    minimum_feasible_satellite_number = None

    for index, configuration in enumerate(
            configurations,
            start=1
    ):

        (
            plane_number,
            satellites_per_plane,
            inclination_degree
        ) = configuration

        total_satellites = (
            plane_number
            * satellites_per_plane
        )

        # 已找到可行方案后，只继续检查相同总卫星数的方案
        if (
            minimum_feasible_satellite_number
            is not None
            and total_satellites
            > minimum_feasible_satellite_number
        ):
            break

        result = evaluate_configuration(
            plane_number=plane_number,
            satellites_per_plane=satellites_per_plane,
            inclination_degree=inclination_degree,
            times=times,
            ground_unit_vectors=ground_unit_vectors
        )

        results.append(result)

        print(
            f"[{index:>3}/"
            f"{total_configuration_number}] "
            f"M={plane_number:>2}, "
            f"N={satellites_per_plane:>2}, "
            f"i={inclination_degree:>4.1f}°, "
            f"总星数={total_satellites:>4}, "
            f"时空覆盖率="
            f"{100 * result['space_time_coverage_ratio']:>7.3f}%, "
            f"最差时刻="
            f"{100 * result['worst_instantaneous_coverage_ratio']:>7.3f}%, "
            f"最大间隙="
            f"{result['maximum_gap_minutes']:>6.1f} min, "
            f"可行={result['feasible']}"
        )

        if result["feasible"]:

            if (
                minimum_feasible_satellite_number
                is None
            ):
                minimum_feasible_satellite_number = (
                    total_satellites
                )

                print(
                    "\n发现第一个粗搜索可行卫星规模："
                    f"{total_satellites} 颗。\n"
                )

    elapsed_time = time.time() - start_time

    sorted_results = sorted(
        results,
        key=ranking_key
    )

    print_top_results(
        sorted_results,
        number=15
    )

    save_results(
        sorted_results
    )

    print(
        f"\n粗搜索计算用时："
        f"{elapsed_time:.2f} s"
    )

    feasible_results = [
        result
        for result in sorted_results
        if result["feasible"]
    ]

    if feasible_results:

        best = feasible_results[0]

        print("\n粗搜索最优可行方案：")

        print(
            f"M = {best['M']}"
        )

        print(
            f"N = {best['N']}"
        )

        print(
            f"i = "
            f"{best['inclination']:.1f}°"
        )

        print(
            f"卫星总数 = "
            f"{best['total_satellites']}"
        )

        print(
            "\n注意：该结果仍需使用更小的"
            "时间步长和空间网格进行精细验证。"
        )

    else:

        print(
            "\n当前粗搜索范围内没有发现"
            "完全连续覆盖方案。"
        )

        print(
            "请查看排名前15的候选方案，"
            "后续围绕它们扩大或细化搜索。"
        )


if __name__ == "__main__":
    main()