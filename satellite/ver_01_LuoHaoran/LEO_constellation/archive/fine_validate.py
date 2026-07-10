import csv
import time

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import cKDTree


# ============================================================
# 1. 基本参数
# ============================================================

# 地球半径，单位 km
R = 6371.0

# 轨道高度，单位 km
h = 550.0

# 轨道半径，单位 km
a = R + h

# 地球引力常数，单位 km^3/s^2
mu = 398600.4418

# 地球自转角速度，单位 rad/s
omega_earth = 7.2921159e-5


# ============================================================
# 2. 待验证的候选星座
# ============================================================

M = 32
N = 64
inclination = 53.75

# Walker 星座相位因子
walker_phase_factor = 21


# ============================================================
# 3. 单星覆盖参数
# ============================================================

# 题目给出的地面覆盖半径，单位 km
coverage_radius = 506.0

# 地心覆盖角，单位 rad
coverage_angle = coverage_radius / R

# 单位球面上对应的弦长
coverage_chord = 2.0 * np.sin(
    coverage_angle / 2.0
)


# ============================================================
# 4. 目标区域
# ============================================================

latitude_min = 4.0
latitude_max = 53.0

longitude_min = 73.0
longitude_max = 135.0


# ============================================================
# 5. 精细仿真参数
# ============================================================

# 经纬度步长，单位 degree
grid_step = 0.5

# 时间步长，单位 s
time_step = 30.0

# 仿真24小时
simulation_duration = 24.0 * 3600.0


# ============================================================
# 6. 构造目标区域网格
# ============================================================

def build_ground_grid():
    """
    构造目标区域经纬度网格。
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

    return (
        latitudes,
        longitudes,
        latitude_grid,
        longitude_grid,
        latitude_rad,
        longitude_rad
    )


# ============================================================
# 7. 计算地面点在ECI坐标系中的单位向量
# ============================================================

def ground_unit_vectors(
        latitude_rad,
        longitude_rad,
        t
):
    """
    考虑地球自转，计算地面点在地心惯性坐标系中的单位向量。
    """

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
# 8. 计算整个星座的卫星位置
# ============================================================

def constellation_unit_vectors(t):
    """
    计算时刻t全部卫星在ECI坐标系中的单位方向向量。
    """

    plane_ids = np.repeat(
        np.arange(M),
        N
    )

    satellite_ids = np.tile(
        np.arange(N),
        M
    )

    # 各轨道面的升交点赤经
    raan = (
        2.0
        * np.pi
        * plane_ids
        / M
    )

    # 轨道面内初始相位
    phase_in_plane = (
        2.0
        * np.pi
        * satellite_ids
        / N
    )

    # 相邻轨道面Walker相位差
    plane_phase = (
        2.0
        * np.pi
        * walker_phase_factor
        * plane_ids
        / (M * N)
    )

    # 轨道平均角速度
    mean_motion = np.sqrt(
        mu / a ** 3
    )

    argument = (
        mean_motion * t
        + phase_in_plane
        + plane_phase
    )

    inclination_rad = np.deg2rad(
        inclination
    )

    # 轨道平面坐标
    x_orbit = np.cos(argument)
    y_orbit = np.sin(argument)

    # 绕x轴旋转轨道倾角
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

    satellite_vectors = np.column_stack(
        (x_eci, y_eci, z_eci)
    )

    # 消除浮点误差，重新归一化
    norms = np.linalg.norm(
        satellite_vectors,
        axis=1,
        keepdims=True
    )

    return satellite_vectors / norms


# ============================================================
# 9. 计算覆盖重数
# ============================================================

def calculate_coverage_counts(
        satellite_vectors,
        ground_vectors
):
    """
    使用KD树计算每个地面点附近的卫星数量。

    在单位球面上，若两个方向向量之间的弦长不超过
    coverage_chord，则卫星覆盖该地面点。
    """

    tree = cKDTree(
        satellite_vectors
    )

    try:
        # 较新版本SciPy支持直接返回邻居数量
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
        # 兼容较旧版本SciPy
        neighbour_lists = tree.query_ball_point(
            ground_vectors,
            r=coverage_chord
        )

        return np.fromiter(
            (
                len(neighbours)
                for neighbours in neighbour_lists
            ),
            dtype=int,
            count=len(neighbour_lists)
        )


# ============================================================
# 10. 计算周期性最大覆盖间隙
# ============================================================

def calculate_point_circular_gap(
        coverage_series
):
    """
    计算某地面点在24小时周期内的最大连续未覆盖步数。

    需要把一天末尾与一天开头连接起来处理。
    """

    time_number = len(
        coverage_series
    )

    if np.all(coverage_series):
        return 0

    if not np.any(coverage_series):
        return time_number

    doubled_series = np.concatenate(
        (
            coverage_series,
            coverage_series
        )
    )

    current_gap = 0
    maximum_gap = 0

    for covered in doubled_series:

        if covered:
            current_gap = 0

        else:
            current_gap += 1

            maximum_gap = max(
                maximum_gap,
                current_gap
            )

    return min(
        maximum_gap,
        time_number
    )


def calculate_all_maximum_gaps(
        coverage_history
):
    """
    计算所有地面点的最大覆盖间隙。
    """

    ground_number = (
        coverage_history.shape[1]
    )

    gap_steps = np.zeros(
        ground_number,
        dtype=int
    )

    for ground_id in range(
            ground_number
    ):

        gap_steps[ground_id] = (
            calculate_point_circular_gap(
                coverage_history[
                    :,
                    ground_id
                ]
            )
        )

    return gap_steps


# ============================================================
# 11. 精细仿真
# ============================================================

def run_fine_validation():
    """
    对候选星座执行24小时精细覆盖验证。
    """

    (
        latitudes,
        longitudes,
        latitude_grid,
        longitude_grid,
        latitude_rad,
        longitude_rad
    ) = build_ground_grid()

    times = np.arange(
        0.0,
        simulation_duration,
        time_step
    )

    time_number = len(times)

    ground_number = (
        latitude_grid.size
    )

    coverage_history = np.zeros(
        (
            time_number,
            ground_number
        ),
        dtype=bool
    )

    total_multiplicity = 0

    print("=" * 60)
    print("开始精细覆盖验证")
    print("=" * 60)

    print(
        f"轨道面数量 M = {M}"
    )

    print(
        f"每轨卫星数量 N = {N}"
    )

    print(
        f"轨道倾角 i = {inclination:.1f}°"
    )

    print(
        f"卫星总数 = {M * N}"
    )

    print(
        f"空间网格步长 = {grid_step:.1f}°"
    )

    print(
        f"时间步长 = {time_step:.0f} s"
    )

    print(
        f"地面网格点数量 = {ground_number}"
    )

    print(
        f"时间采样点数量 = {time_number}"
    )

    print("=" * 60)

    start_time = time.time()

    for time_id, t in enumerate(times):

        satellite_vectors = (
            constellation_unit_vectors(t)
        )

        ground_vectors = (
            ground_unit_vectors(
                latitude_rad,
                longitude_rad,
                t
            )
        )

        coverage_counts = (
            calculate_coverage_counts(
                satellite_vectors,
                ground_vectors
            )
        )

        coverage_history[time_id] = (
            coverage_counts >= 1
        )

        total_multiplicity += np.sum(
            coverage_counts
        )

        if (
            time_id % 60 == 0
            or time_id == time_number - 1
        ):

            progress = (
                100.0
                * (time_id + 1)
                / time_number
            )

            print(
                f"仿真进度："
                f"{progress:6.2f}%"
            )

    elapsed_time = (
        time.time() - start_time
    )

    # 时空覆盖率
    space_time_coverage_ratio = np.mean(
        coverage_history
    )

    # 每个时刻整个区域是否全部覆盖
    fully_covered_state = np.all(
        coverage_history,
        axis=1
    )

    full_region_coverage_ratio = np.mean(
        fully_covered_state
    )

    # 每个时刻的区域覆盖比例
    instantaneous_coverage_ratio = np.mean(
        coverage_history,
        axis=1
    )

    worst_instantaneous_ratio = np.min(
        instantaneous_coverage_ratio
    )

    worst_time_id = np.argmin(
        instantaneous_coverage_ratio
    )

    worst_time_minutes = (
        times[worst_time_id]
        / 60.0
    )

    # 每个地点24小时内的时间覆盖率
    point_time_coverage_ratio = np.mean(
        coverage_history,
        axis=0
    )

    minimum_point_ratio = np.min(
        point_time_coverage_ratio
    )

    worst_point_id = np.argmin(
        point_time_coverage_ratio
    )

    worst_point_latitude = (
        latitude_grid.ravel()[
            worst_point_id
        ]
    )

    worst_point_longitude = (
        longitude_grid.ravel()[
            worst_point_id
        ]
    )

    # 平均覆盖重数
    average_multiplicity = (
        total_multiplicity
        / (
            time_number
            * ground_number
        )
    )

    # 最大覆盖间隙
    gap_steps = (
        calculate_all_maximum_gaps(
            coverage_history
        )
    )

    maximum_gap_steps = np.max(
        gap_steps
    )

    maximum_gap_minutes = (
        maximum_gap_steps
        * time_step
        / 60.0
    )

    maximum_gap_point_id = np.argmax(
        gap_steps
    )

    maximum_gap_latitude = (
        latitude_grid.ravel()[
            maximum_gap_point_id
        ]
    )

    maximum_gap_longitude = (
        longitude_grid.ravel()[
            maximum_gap_point_id
        ]
    )

    feasible = (
        np.all(coverage_history)
        and maximum_gap_steps == 0
    )

    results = {
        "M": M,
        "N": N,
        "inclination": inclination,
        "total_satellites": M * N,

        "grid_step_degree": grid_step,
        "time_step_second": time_step,

        "space_time_coverage_ratio":
            space_time_coverage_ratio,

        "full_region_coverage_ratio":
            full_region_coverage_ratio,

        "worst_instantaneous_ratio":
            worst_instantaneous_ratio,

        "worst_time_minutes":
            worst_time_minutes,

        "minimum_point_ratio":
            minimum_point_ratio,

        "worst_point_latitude":
            worst_point_latitude,

        "worst_point_longitude":
            worst_point_longitude,

        "average_multiplicity":
            average_multiplicity,

        "maximum_gap_minutes":
            maximum_gap_minutes,

        "maximum_gap_latitude":
            maximum_gap_latitude,

        "maximum_gap_longitude":
            maximum_gap_longitude,

        "feasible":
            feasible
    }

    point_coverage_grid = (
        100.0
        * point_time_coverage_ratio.reshape(
            latitude_grid.shape
        )
    )

    return (
        results,
        point_coverage_grid,
        latitude_grid,
        longitude_grid,
        elapsed_time
    )


# ============================================================
# 12. 打印结果
# ============================================================

def print_results(
        results,
        elapsed_time
):
    """
    打印精细验证结果。
    """

    print("\n")
    print("=" * 70)
    print("候选星座精细验证结果")
    print("=" * 70)

    print(
        f"M = {results['M']}"
    )

    print(
        f"N = {results['N']}"
    )

    print(
        f"i = "
        f"{results['inclination']:.1f}°"
    )

    print(
        f"卫星总数 = "
        f"{results['total_satellites']}"
    )

    print("-" * 70)

    print(
        "时空覆盖率："
        f"{100.0 * results['space_time_coverage_ratio']:.6f}%"
    )

    print(
        "全区域连续覆盖时间比例："
        f"{100.0 * results['full_region_coverage_ratio']:.6f}%"
    )

    print(
        "最差时刻区域覆盖率："
        f"{100.0 * results['worst_instantaneous_ratio']:.6f}%"
    )

    print(
        "最差时刻："
        f"{results['worst_time_minutes']:.2f} min"
    )

    print(
        "最差地点时间覆盖率："
        f"{100.0 * results['minimum_point_ratio']:.6f}%"
    )

    print(
        "最差地点："
        f"纬度 {results['worst_point_latitude']:.2f}°，"
        f"经度 {results['worst_point_longitude']:.2f}°"
    )

    print(
        "平均覆盖重数："
        f"{results['average_multiplicity']:.6f}"
    )

    print(
        "最大覆盖间隙："
        f"{results['maximum_gap_minutes']:.2f} min"
    )

    print(
        "最大间隙地点："
        f"纬度 {results['maximum_gap_latitude']:.2f}°，"
        f"经度 {results['maximum_gap_longitude']:.2f}°"
    )

    print("-" * 70)

    print(
        f"精细验证是否通过："
        f"{results['feasible']}"
    )

    print(
        f"计算用时："
        f"{elapsed_time:.2f} s"
    )

    print("=" * 70)


# ============================================================
# 13. 保存结果
# ============================================================

def save_summary(results):
    """
    保存精细验证汇总结果。
    """

    filename = (
        "fine_validation_summary.csv"
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
                results.keys()
            )
        )

        writer.writeheader()

        writer.writerow(
            results
        )

    print(
        f"结果已保存到：{filename}"
    )


def draw_heatmap(
        point_coverage_grid
):
    """
    绘制各地面网格点的24小时时间覆盖率。
    """

    plt.figure(
        figsize=(10, 7)
    )

    image = plt.imshow(
        point_coverage_grid,
        origin="lower",
        extent=[
            longitude_min,
            longitude_max,
            latitude_min,
            latitude_max
        ],
        aspect="auto",
        vmin=0.0,
        vmax=100.0
    )

    plt.colorbar(
        image,
        label="Time coverage ratio (%)"
    )

    plt.xlabel(
        "Longitude (degree)"
    )

    plt.ylabel(
        "Latitude (degree)"
    )

    plt.title(
        "Fine Validation of Regional Coverage"
    )

    plt.tight_layout()

    filename = (
        "fine_validation_heatmap.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    print(
        f"覆盖热力图已保存到：{filename}"
    )

    plt.show()


# ============================================================
# 14. 主程序
# ============================================================

def main():

    (
        results,
        point_coverage_grid,
        latitude_grid,
        longitude_grid,
        elapsed_time
    ) = run_fine_validation()

    print_results(
        results,
        elapsed_time
    )

    save_summary(
        results
    )

    draw_heatmap(
        point_coverage_grid
    )


if __name__ == "__main__":
    main()