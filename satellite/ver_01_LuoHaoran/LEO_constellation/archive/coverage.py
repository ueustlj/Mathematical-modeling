import numpy as np

from archive.parameter import (
    R,
    omega_earth,
    coverage_angle,
    latitude_min,
    latitude_max,
    longitude_min,
    longitude_max,
    grid_step,
    time_step,
    simulation_duration,
    M,
    N,
    inclination
)

from archive.orbit import constellation_positions


def create_ground_grid():
    """
    在目标区域建立规则经纬度网格。

    返回
    ----------
    latitudes : 一维纬度数组
    longitudes : 一维经度数组
    latitude_grid : 二维纬度网格
    longitude_grid : 二维经度网格
    """

    latitudes = np.arange(
        latitude_min,
        latitude_max + grid_step,
        grid_step
    )

    longitudes = np.arange(
        longitude_min,
        longitude_max + grid_step,
        grid_step
    )

    longitude_grid, latitude_grid = np.meshgrid(
        longitudes,
        latitudes
    )

    return (
        latitudes,
        longitudes,
        latitude_grid,
        longitude_grid
    )


def ground_positions_eci(
        latitude_grid,
        longitude_grid,
        t
):
    """
    将地面网格点转换为地心惯性坐标 ECI。

    这里考虑地球自转：
    地面经度随时间在惯性系中变化。
    """

    latitude_rad = np.deg2rad(
        latitude_grid.ravel()
    )

    longitude_rad = np.deg2rad(
        longitude_grid.ravel()
    )

    # 地球向东自转
    longitude_eci = (
        longitude_rad
        + omega_earth * t
    )

    x = (
        R
        * np.cos(latitude_rad)
        * np.cos(longitude_eci)
    )

    y = (
        R
        * np.cos(latitude_rad)
        * np.sin(longitude_eci)
    )

    z = (
        R
        * np.sin(latitude_rad)
    )

    return np.column_stack(
        (x, y, z)
    )


def calculate_coverage_counts(
        satellite_positions,
        ground_positions
):
    """
    计算每个地面点同时被多少颗卫星覆盖。

    判定条件：
    卫星星下点与地面点的地心夹角
    不超过最大覆盖地心角。
    """

    satellite_norm = np.linalg.norm(
        satellite_positions,
        axis=1,
        keepdims=True
    )

    ground_norm = np.linalg.norm(
        ground_positions,
        axis=1,
        keepdims=True
    )

    satellite_unit = (
        satellite_positions
        / satellite_norm
    )

    ground_unit = (
        ground_positions
        / ground_norm
    )

    # 每一行代表一个地面点
    # 每一列代表一颗卫星
    cosine_matrix = (
        ground_unit
        @ satellite_unit.T
    )

    cosine_limit = np.cos(
        coverage_angle
    )

    visible_matrix = (
        cosine_matrix
        >= cosine_limit
    )

    coverage_counts = np.sum(
        visible_matrix,
        axis=1
    )

    return coverage_counts


def evaluate_constellation(
        plane_number=M,
        satellites_per_plane=N,
        inclination_degree=inclination
):
    """
    对整个星座执行24小时覆盖仿真。

    返回指标：
    1. 时空覆盖率
    2. 全区域连续覆盖时间比例
    3. 平均覆盖重数
    4. 最大覆盖间隙
    5. 各地面点的时间覆盖率
    """

    (
        latitudes,
        longitudes,
        latitude_grid,
        longitude_grid
    ) = create_ground_grid()

    ground_point_number = (
        latitude_grid.size
    )

    times = np.arange(
        0.0,
        simulation_duration,
        time_step
    )

    time_number = len(times)

    total_covered_number = 0

    total_coverage_multiplicity = 0

    fully_covered_time_number = 0

    covered_time_per_point = np.zeros(
        ground_point_number,
        dtype=int
    )

    current_gap_steps = np.zeros(
        ground_point_number,
        dtype=int
    )

    maximum_gap_steps = np.zeros(
        ground_point_number,
        dtype=int
    )

    for index, t in enumerate(times):

        satellite_positions = (
            constellation_positions(
                t=t,
                plane_number=plane_number,
                satellites_per_plane=satellites_per_plane,
                inclination_degree=inclination_degree
            )
        )

        ground_positions = (
            ground_positions_eci(
                latitude_grid,
                longitude_grid,
                t
            )
        )

        coverage_counts = (
            calculate_coverage_counts(
                satellite_positions,
                ground_positions
            )
        )

        covered = coverage_counts >= 1

        total_covered_number += np.sum(
            covered
        )

        total_coverage_multiplicity += np.sum(
            coverage_counts
        )

        covered_time_per_point += covered.astype(
            int
        )

        # 判断当前时刻是否整个区域全部覆盖
        if np.all(covered):

            fully_covered_time_number += 1

        # 更新连续未覆盖时间
        current_gap_steps[covered] = 0

        current_gap_steps[~covered] += 1

        maximum_gap_steps = np.maximum(
            maximum_gap_steps,
            current_gap_steps
        )

        # 显示计算进度
        if index % 30 == 0:

            progress = (
                100.0
                * index
                / time_number
            )

            print(
                f"仿真进度：{progress:.1f}%"
            )

    space_time_coverage_ratio = (
        total_covered_number
        / (
            time_number
            * ground_point_number
        )
    )

    full_region_coverage_ratio = (
        fully_covered_time_number
        / time_number
    )

    average_multiplicity = (
        total_coverage_multiplicity
        / (
            time_number
            * ground_point_number
        )
    )

    maximum_gap_minutes = (
        np.max(maximum_gap_steps)
        * time_step
        / 60.0
    )

    point_time_coverage_ratio = (
        covered_time_per_point
        / time_number
    )

    point_time_coverage_grid = (
        point_time_coverage_ratio.reshape(
            latitude_grid.shape
        )
    )

    return {
        "plane_number": plane_number,

        "satellites_per_plane":
            satellites_per_plane,

        "total_satellites":
            plane_number
            * satellites_per_plane,

        "inclination":
            inclination_degree,

        "space_time_coverage_ratio":
            space_time_coverage_ratio,

        "full_region_coverage_ratio":
            full_region_coverage_ratio,

        "average_multiplicity":
            average_multiplicity,

        "maximum_gap_minutes":
            maximum_gap_minutes,

        "latitude_grid":
            latitude_grid,

        "longitude_grid":
            longitude_grid,

        "point_time_coverage_grid":
            point_time_coverage_grid
    }