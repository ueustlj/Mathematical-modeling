import csv
import time

import numpy as np
from scipy.spatial import cKDTree


# ============================================================
# 基本参数
# ============================================================

R = 6371.0
h = 550.0
a = R + h

mu = 398600.4418
omega_earth = 7.2921159e-5

M = 32
N = 64

coverage_radius = 506.0
coverage_angle = coverage_radius / R
coverage_chord = 2.0 * np.sin(coverage_angle / 2.0)


# ============================================================
# 目标区域与精细仿真参数
# ============================================================

latitude_min = 4.0
latitude_max = 53.0

longitude_min = 73.0
longitude_max = 135.0

grid_step = 1.0
time_step = 60.0
simulation_duration = 24.0 * 3600.0


# ============================================================
# 局部搜索范围
# ============================================================

# 在54°附近细调
INCLINATION_VALUES = np.arange(
    53.0,
    55.0 + 0.001,
    0.25
)

# 取中等精度搜索中表现较好的相位因子
PHASE_FACTOR_VALUES = [
    0,
    19,
    21,
    23,
    28,
    30
]


# ============================================================
# 建立地面网格
# ============================================================

def build_ground_grid():

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

    return (
        times,
        latitude_grid,
        longitude_grid,
        latitude_rad,
        longitude_rad
    )


def ground_vectors_at_time(
        latitude_rad,
        longitude_rad,
        t
):

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

    return np.column_stack(
        (x, y, z)
    )


# ============================================================
# 星座模型
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


def constellation_vectors(
        t,
        inclination_degree,
        phase_factor
):

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

    x_inclined = x_orbit

    y_inclined = (
        y_orbit
        * np.cos(inclination_rad)
    )

    z_inclined = (
        y_orbit
        * np.sin(inclination_rad)
    )

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

    vectors /= np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    return vectors


# ============================================================
# 覆盖判断
# ============================================================

def calculate_coverage_counts(
        satellite_vectors,
        ground_vectors
):

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
# 周期最大空窗
# ============================================================

def calculate_circular_maximum_gap(
        coverage_history
):

    time_number, ground_number = (
        coverage_history.shape
    )

    overall_maximum = 0

    for ground_id in range(
            ground_number
    ):

        series = coverage_history[
            :,
            ground_id
        ]

        if np.all(series):
            point_maximum = 0

        elif not np.any(series):
            point_maximum = time_number

        else:

            doubled = np.concatenate(
                (series, series)
            )

            current_gap = 0
            point_maximum = 0

            for covered in doubled:

                if covered:
                    current_gap = 0

                else:
                    current_gap += 1

                    point_maximum = max(
                        point_maximum,
                        current_gap
                    )

            point_maximum = min(
                point_maximum,
                time_number
            )

        overall_maximum = max(
            overall_maximum,
            point_maximum
        )

    return (
        overall_maximum
        * time_step
        / 60.0
    )


# ============================================================
# 评价一个候选方案
# ============================================================

def evaluate_configuration(
        inclination_degree,
        phase_factor,
        times,
        latitude_rad,
        longitude_rad
):

    time_number = len(times)
    ground_number = len(latitude_rad)

    coverage_history = np.zeros(
        (
            time_number,
            ground_number
        ),
        dtype=bool
    )

    total_multiplicity = 0

    for time_id, t in enumerate(times):

        satellite_vectors = (
            constellation_vectors(
                t,
                inclination_degree,
                phase_factor
            )
        )

        ground_vectors = (
            ground_vectors_at_time(
                latitude_rad,
                longitude_rad,
                t
            )
        )

        counts = calculate_coverage_counts(
            satellite_vectors,
            ground_vectors
        )

        coverage_history[time_id] = (
            counts >= 1
        )

        total_multiplicity += np.sum(
            counts
        )

    space_time_ratio = np.mean(
        coverage_history
    )

    full_region_state = np.all(
        coverage_history,
        axis=1
    )

    full_region_ratio = np.mean(
        full_region_state
    )

    instantaneous_ratio = np.mean(
        coverage_history,
        axis=1
    )

    worst_instantaneous_ratio = np.min(
        instantaneous_ratio
    )

    point_time_ratio = np.mean(
        coverage_history,
        axis=0
    )

    minimum_point_ratio = np.min(
        point_time_ratio
    )

    average_multiplicity = (
        total_multiplicity
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

    total_uncovered_cells = (
        coverage_history.size
        - np.count_nonzero(
            coverage_history
        )
    )

    maximum_uncovered_points = np.max(
        ground_number
        - np.sum(
            coverage_history,
            axis=1
        )
    )

    feasible = bool(
        np.all(coverage_history)
    )

    return {
        "inclination":
            float(inclination_degree),

        "phase_factor":
            int(phase_factor),

        "total_satellites":
            M * N,

        "space_time_coverage_ratio":
            space_time_ratio,

        "full_region_coverage_ratio":
            full_region_ratio,

        "worst_instantaneous_ratio":
            worst_instantaneous_ratio,

        "minimum_point_coverage_ratio":
            minimum_point_ratio,

        "average_multiplicity":
            average_multiplicity,

        "maximum_gap_minutes":
            maximum_gap_minutes,

        "total_uncovered_cells":
            int(total_uncovered_cells),

        "maximum_uncovered_points":
            int(maximum_uncovered_points),

        "feasible":
            feasible
    }


def ranking_key(result):

    return (
        0 if result["feasible"] else 1,

        -result[
            "full_region_coverage_ratio"
        ],

        -result[
            "worst_instantaneous_ratio"
        ],

        result[
            "total_uncovered_cells"
        ],

        result[
            "maximum_gap_minutes"
        ]
    )


# ============================================================
# 主程序
# ============================================================

def main():

    (
        times,
        latitude_grid,
        longitude_grid,
        latitude_rad,
        longitude_rad
    ) = build_ground_grid()

    configurations = [
        (inclination_degree, phase_factor)

        for inclination_degree
        in INCLINATION_VALUES

        for phase_factor
        in PHASE_FACTOR_VALUES
    ]

    print("=" * 60)

    print(
        f"卫星总数：{M * N}"
    )

    print(
        f"地面网格点数："
        f"{latitude_grid.size}"
    )

    print(
        f"时间点数量："
        f"{len(times)}"
    )

    print(
        f"待测试方案数："
        f"{len(configurations)}"
    )

    print("=" * 60)

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
            inclination_degree,
            phase_factor,
            times,
            latitude_rad,
            longitude_rad
        )

        results.append(result)

        print(
            f"[{index:>2}/"
            f"{len(configurations)}] "
            f"i={inclination_degree:>5.2f}°, "
            f"F={phase_factor:>2}, "
            f"时空覆盖率="
            f"{100 * result['space_time_coverage_ratio']:.5f}%, "
            f"全区域时间比例="
            f"{100 * result['full_region_coverage_ratio']:.3f}%, "
            f"最大未覆盖点数="
            f"{result['maximum_uncovered_points']:>3}, "
            f"最大空窗="
            f"{result['maximum_gap_minutes']:.1f} min, "
            f"可行="
            f"{result['feasible']}"
        )

    sorted_results = sorted(
        results,
        key=ranking_key
    )

    filename = (
        "local_refine_results.csv"
    )

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                sorted_results[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(sorted_results)

    print("\n")
    print("=" * 110)

    print(
        "排名  倾角    F   时空覆盖率   "
        "全区域时间比例   最差时刻覆盖率   "
        "总漏点数   最大未覆盖点数   最大空窗   可行"
    )

    print("=" * 110)

    for rank, result in enumerate(
            sorted_results[:15],
            start=1
    ):

        print(
            f"{rank:>4} "
            f"{result['inclination']:>6.2f} "
            f"{result['phase_factor']:>3} "
            f"{100 * result['space_time_coverage_ratio']:>11.5f}% "
            f"{100 * result['full_region_coverage_ratio']:>16.3f}% "
            f"{100 * result['worst_instantaneous_ratio']:>16.5f}% "
            f"{result['total_uncovered_cells']:>10} "
            f"{result['maximum_uncovered_points']:>16} "
            f"{result['maximum_gap_minutes']:>9.1f} "
            f"{str(result['feasible']):>7}"
        )

    elapsed_time = (
        time.time()
        - start_time
    )

    best = sorted_results[0]

    print("=" * 110)

    print(
        f"\n当前最佳："
        f"i={best['inclination']:.2f}°，"
        f"F={best['phase_factor']}"
    )

    print(
        f"是否精细可行："
        f"{best['feasible']}"
    )

    print(
        f"结果文件：{filename}"
    )

    print(
        f"计算用时："
        f"{elapsed_time:.2f} s"
    )


if __name__ == "__main__":
    main()