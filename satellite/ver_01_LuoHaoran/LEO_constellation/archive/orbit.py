import numpy as np

from archive.parameter import (
    a,
    mu,
    M,
    N,
    inclination,
    walker_phase_factor
)


def orbital_period():
    """
    计算圆轨道周期，单位 s
    """

    return 2.0 * np.pi * np.sqrt(a ** 3 / mu)


def satellite_position(
        t,
        plane_id,
        satellite_id,
        plane_number=M,
        satellites_per_plane=N,
        inclination_degree=inclination
):
    """
    计算单颗卫星在地心惯性坐标系 ECI 中的位置。

    参数
    ----------
    t : float
        仿真时间，单位 s

    plane_id : int
        轨道面编号，从 0 开始

    satellite_id : int
        轨道面内卫星编号，从 0 开始

    plane_number : int
        轨道面数量

    satellites_per_plane : int
        每个轨道面的卫星数量

    inclination_degree : float
        轨道倾角，单位 degree

    返回
    ----------
    numpy.ndarray
        卫星三维坐标 [x, y, z]，单位 km
    """

    # 轨道平均角速度
    mean_motion = np.sqrt(mu / a ** 3)

    # 第 plane_id 个轨道面的升交点赤经
    raan = (
        2.0
        * np.pi
        * plane_id
        / plane_number
    )

    # 同一轨道面内卫星均匀分布
    phase_in_plane = (
        2.0
        * np.pi
        * satellite_id
        / satellites_per_plane
    )

    # 相邻轨道面之间加入 Walker 相位差
    plane_phase = (
        2.0
        * np.pi
        * walker_phase_factor
        * plane_id
        / (plane_number * satellites_per_plane)
    )

    # 卫星在轨道平面中的幅角
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

    # 绕 x 轴旋转轨道倾角
    x_inclined = x_orbit

    y_inclined = (
        y_orbit
        * np.cos(inclination_rad)
    )

    z_inclined = (
        y_orbit
        * np.sin(inclination_rad)
    )

    # 绕 z 轴旋转升交点赤经
    x_eci = (
        x_inclined * np.cos(raan)
        - y_inclined * np.sin(raan)
    )

    y_eci = (
        x_inclined * np.sin(raan)
        + y_inclined * np.cos(raan)
    )

    z_eci = z_inclined

    return np.array(
        [x_eci, y_eci, z_eci],
        dtype=float
    )


def constellation_positions(
        t,
        plane_number=M,
        satellites_per_plane=N,
        inclination_degree=inclination
):
    """
    计算某一时刻整个星座中所有卫星的位置。

    返回数组形状：
    [卫星总数, 3]
    """

    positions = []

    for plane_id in range(plane_number):

        for satellite_id in range(
                satellites_per_plane
        ):

            position = satellite_position(
                t=t,
                plane_id=plane_id,
                satellite_id=satellite_id,
                plane_number=plane_number,
                satellites_per_plane=satellites_per_plane,
                inclination_degree=inclination_degree
            )

            positions.append(position)

    return np.array(positions)