import csv
import time

import numpy as np
from scipy.spatial import cKDTree


# ============================================================
# 1. 地球与轨道参数
# ============================================================

R = 6371.0
h = 550.0
a = R + h

mu = 398600.4418

omega_earth = 7.2921159e-5


# ============================================================
# 2. 固定星座规模
# ============================================================

M = 32
N = 64

TOTAL_SATELLITES = M * N


# ============================================================
# 3. 搜索变量
# ============================================================

# 在粗搜索最优倾角56°附近细化
INCLINATION_VALUES = np.arange(
    54.0,
    58.0 + 0.01,
    0.5
)

# Walker Delta星座相位因子
# 通常取 0 到 M-1 的整数
PHASE_FACTOR_VALUES = range(M)


# ============================================================
# 4. 覆盖参数
# ============================================================

coverage_radius = 506.0

coverage_angle = coverage_radius / R

# 单位球上的覆盖弦长
coverage_chord = 2.0 * np.sin(
    coverage_angle / 2.0
)


# ============================================================
# 5. 目标区域
# ============================================================

latitude_min = 4.0
latitude_max = 53.0

longitude_min = 73.0
longitude_max = 135.0


# ============================================================
# 6. 搜索精度
# ============================================================

# 先用中等精度筛选相位
grid_step = 2.0

# 每2分钟采样一次
time_step = 120.0

simulation_duration = 24.0 * 3600.0


# ============================================================
# 7. 构造地面网格
# ============================================================

def build_ground_vectors():
    """
    预先计算全部时刻的地面网格单位向量。
    """

    latitudes = np.arange(
        latitude_min,
        latitude_max + 0.5 * grid_step,
        grid_step
    )

    longitudes = np.arange(
        longitude_min,
        longitude_max + 0.5 * grid_step,
        grid_step
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
        simulation_duration,
        time_step
    )

    ground_vectors = []

    for t in times:

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

        ground_vectors.append(
            np.column_stack((x, y, z))
        )

    return (
        times,
        np.asarray(ground_vectors),
        latitude_grid,
        longitude_grid
    )


# ============================================================
# 8. 卫星星座单位向量
# ============================================================

plane_ids = np.repeat(
    np.arange(M),
    N
)

satellite_ids = np.tile(
    np.arange(N),
    M
)

raan = (
    2.0
    * np.pi
    * plane_ids
    / M
)

phase_in_plane = (
    2.0
    * np.pi
    * satellite_ids
    / N
)

mean_motion = np.sqrt(
    mu / a ** 3
)


def constellation_unit_vectors(
        t,
        inclination_degree,
        phase_factor
):
    """
    计算指定倾角、相位因子和时刻下，
    全部卫星的ECI单位方向向量。
    """

    plane_phase = (
        2.0
        * np.pi
        * phase_factor
        * plane_ids
        / (M * N)
    )

    argument = (
        mean_motion * t
        + phase_in_plane
        + plane_phase
    )

    inclination_rad = np.deg2rad(
        inclination_degree
    )

    x_orbit = np.cos(argument)
    y_orbit = np.sin(argument)

    # 绕x轴旋转倾角
    x_inclined = x_orbit

    y_inclined = (
        y_orbit
        * np.cos(inclination_rad)
    )

    z_inclined = (
        y_orbit
        * np.sin(inclination_rad)
    )

    # 绕z轴旋转升交点赤经
    x_eci = (
        x_inclined * np.cos(raan)
        - y_inclined * np.sin(raan)
    )

    y_eci = (
        x_inclined * np.sin(raan)
        + y_inclined * np.cos(raan)
    )

    z_eci = z_inclined

    vectors = np.column_stack(
        (x_eci, y_eci, z_eci)
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    return vectors / norms


# ============================================================
# 9. KD树覆盖计数
# ============================================================

def coverage_counts(
        satellite_vectors,
        ground_vectors
):
    """
    计算每个地面网格点被多少颗卫星覆盖。
    """

    tree = cKDTree(
        satellite_vectors
    )

    try:
        counts = tree.query_ball_point(
            ground_vectors,
            r=coverage_chord,
            return_length=True
        )

        return np.asarray(
            counts,
            dtype=int
        )

    except TypeError:

        neighbours = tree.query_ball_point(
            ground_vectors,
            r=coverage_chord
        )

        return np.fromiter(
            (
                len(item)
                for item in neighbours
            ),
            dtype=int,
            count=len(neighbours)
        )


# ============================================================
# 10. 评价一种相位布局
# ============================================================

def evaluate_configuration(
        inclination_degree,
        phase_factor,
        times,
        all_ground_vectors
):
    """
    计算给定倾角和相位因子的覆盖性能。
    """

    time_number = len(times)

    ground_number = (
        all_ground_vectors.shape[1]
    )

    total_covered = 0
    total_multiplicity = 0

    full_region_time_count = 0

    worst_instantaneous_ratio = 1.0

    point_covered_count = np.zeros(
        ground_number,
        dtype=int
    )

    # 记录连续未覆盖步数
    current_gap = np.zeros(
        ground_number,
        dtype=int
    )

    maximum_gap = np.zeros(
        ground_number,
        dtype=int
    )

    for time_id, t in enumerate(times):

        satellite_vectors = (
            constellation_unit_vectors(
                t=t,
                inclination_degree=inclination_degree,
                phase_factor=phase_factor
            )
        )

        counts = coverage_counts(
            satellite_vectors,
            all_ground_vectors[time_id]
        )

        covered = counts >= 1

        covered_number = np.sum(
            covered
        )

        instantaneous_ratio = (
            covered_number
            / ground_number
        )

        worst_instantaneous_ratio = min(
            worst_instantaneous_ratio,
            instantaneous_ratio
        )

        total_covered += covered_number

        total_multiplicity += np.sum(
            counts
        )

        point_covered_count += covered.astype(
            int
        )

        if np.all(covered):

            full_region_time_count += 1

        current_gap[covered] = 0

        current_gap[~covered] += 1

        maximum_gap = np.maximum(
            maximum_gap,
            current_gap
        )

    space_time_coverage_ratio = (
        total_covered
        / (
            time_number
            * ground_number
        )
    )

    full_region_coverage_ratio = (
        full_region_time_count
        / time_number
    )

    point_time_coverage_ratio = (
        point_covered_count
        / time_number
    )

    minimum_point_coverage_ratio = np.min(
        point_time_coverage_ratio
    )

    average_multiplicity = (
        total_multiplicity
        / (
            time_number
            * ground_number
        )
    )

    maximum_gap_minutes = (
        np.max(maximum_gap)
        * time_step
        / 60.0
    )

    feasible = (
        full_region_time_count
        == time_number
    )

    return {
        "M": M,

        "N": N,

        "total_satellites":
            TOTAL_SATELLITES,

        "inclination":
            float(inclination_degree),

        "phase_factor":
            int(phase_factor),

        "space_time_coverage_ratio":
            space_time_coverage_ratio,

        "full_region_coverage_ratio":
            full_region_coverage_ratio,

        "worst_instantaneous_ratio":
            worst_instantaneous_ratio,

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
# 11. 排序函数
# ============================================================

def ranking_key(result):
    """
    优先级：
    1. 可行方案优先；
    2. 全区域覆盖时间比例；
    3. 最差时刻覆盖率；
    4. 最差地点覆盖率；
    5. 时空覆盖率；
    6. 最大覆盖间隙。
    """

    return (
        0 if result["feasible"] else 1,

        -result[
            "full_region_coverage_ratio"
        ],

        -result[
            "worst_instantaneous_ratio"
        ],

        -result[
            "minimum_point_coverage_ratio"
        ],

        -result[
            "space_time_coverage_ratio"
        ],

        result[
            "maximum_gap_minutes"
        ]
    )


# ============================================================
# 12. 保存结果
# ============================================================

def save_results(results):
    """
    保存全部搜索结果。
    """

    filename = "phase_search_results.csv"

    fieldnames = list(
        results[0].keys()
    )

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
        f"\n全部结果已保存至：{filename}"
    )


# ============================================================
# 13. 打印前15名
# ============================================================

def print_top_results(
        results,
        number=15
):
    """
    输出性能最好的候选方案。
    """

    print("\n")

    print("=" * 125)

    print(
        "排名  倾角    F   "
        "时空覆盖率   全区域覆盖时间比例   "
        "最差时刻覆盖率   最差地点覆盖率   "
        "平均覆盖重数   最大间隙/min   可行"
    )

    print("=" * 125)

    for rank, result in enumerate(
            results[:number],
            start=1
    ):

        print(
            f"{rank:>4} "
            f"{result['inclination']:>6.1f} "
            f"{result['phase_factor']:>4} "
            f"{100 * result['space_time_coverage_ratio']:>11.4f}% "
            f"{100 * result['full_region_coverage_ratio']:>18.4f}% "
            f"{100 * result['worst_instantaneous_ratio']:>15.4f}% "
            f"{100 * result['minimum_point_coverage_ratio']:>15.4f}% "
            f"{result['average_multiplicity']:>14.4f} "
            f"{result['maximum_gap_minutes']:>13.2f} "
            f"{str(result['feasible']):>7}"
        )

    print("=" * 125)


# ============================================================
# 14. 主程序
# ============================================================

def main():

    print("正在生成目标区域地面网格……")

    (
        times,
        all_ground_vectors,
        latitude_grid,
        longitude_grid
    ) = build_ground_vectors()

    configurations = []

    for inclination_degree in (
            INCLINATION_VALUES
    ):

        for phase_factor in (
                PHASE_FACTOR_VALUES
        ):

            configurations.append(
                (
                    inclination_degree,
                    phase_factor
                )
            )

    configuration_number = len(
        configurations
    )

    print(
        f"地面网格点数："
        f"{latitude_grid.size}"
    )

    print(
        f"时间采样点数："
        f"{len(times)}"
    )

    print(
        f"共搜索："
        f"{configuration_number} "
        f"组倾角—相位组合"
    )

    print("\n开始搜索……\n")

    start_time = time.time()

    results = []

    for index, (
            inclination_degree,
            phase_factor
    ) in enumerate(
            configurations,
            start=1
    ):

        result = evaluate_configuration(
            inclination_degree=
                inclination_degree,

            phase_factor=
                phase_factor,

            times=
                times,

            all_ground_vectors=
                all_ground_vectors
        )

        results.append(result)

        print(
            f"[{index:>3}/"
            f"{configuration_number}] "
            f"i="
            f"{inclination_degree:>4.1f}°, "
            f"F="
            f"{phase_factor:>2}, "
            f"时空覆盖率="
            f"{100 * result['space_time_coverage_ratio']:>8.4f}%, "
            f"全区域时间比例="
            f"{100 * result['full_region_coverage_ratio']:>8.4f}%, "
            f"最差时刻="
            f"{100 * result['worst_instantaneous_ratio']:>8.4f}%, "
            f"最大间隙="
            f"{result['maximum_gap_minutes']:>5.1f} min, "
            f"可行="
            f"{result['feasible']}"
        )

    sorted_results = sorted(
        results,
        key=ranking_key
    )

    elapsed_time = (
        time.time()
        - start_time
    )

    print_top_results(
        sorted_results
    )

    save_results(
        sorted_results
    )

    print(
        f"\n相位搜索用时："
        f"{elapsed_time:.2f} s"
    )

    best = sorted_results[0]

    print("\n当前最优候选：")

    print(
        f"倾角 i = "
        f"{best['inclination']:.1f}°"
    )

    print(
        f"相位因子 F = "
        f"{best['phase_factor']}"
    )

    print(
        "全区域连续覆盖时间比例 = "
        f"{100 * best['full_region_coverage_ratio']:.4f}%"
    )

    print(
        "最差时刻区域覆盖率 = "
        f"{100 * best['worst_instantaneous_ratio']:.4f}%"
    )

    print(
        "最大覆盖间隙 = "
        f"{best['maximum_gap_minutes']:.2f} min"
    )

    print(
        "是否在当前搜索精度下可行 = "
        f"{best['feasible']}"
    )


if __name__ == "__main__":
    main()