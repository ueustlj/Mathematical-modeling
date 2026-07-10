"""
1.生成目标区域网格：把中国地面对应点离散成很多地面点
2.生成星座
3.计算每颗卫星的星下点
4.判断覆盖
5.统计是否连续覆盖
"""



import numpy as np

# =========================
# 1. 基本常数
# =========================

R_E = 6371.0          # 地球半径，km
H = 550.0             # 轨道高度，km
A = R_E + H           # 轨道半径，km
MU = 398600.4418      # 地球引力常数，km^3/s^2
OMEGA_E = 7.2921159e-5  # 地球自转角速度，rad/s

COVER_RADIUS = 506.0  # 单星地面覆盖半径，km
COVER_ANGLE = COVER_RADIUS / R_E  # 覆盖地心角，rad
COS_COVER = np.cos(COVER_ANGLE)

# 目标区域
LAT_MIN, LAT_MAX = 4.0, 53.0
LON_MIN, LON_MAX = 73.0, 135.0


# =========================
# 2. 工具函数
# =========================

def deg2rad(x):
    return np.deg2rad(x)


def rad2deg(x):
    return np.rad2deg(x)


def wrap_lon_deg(lon):
    """
    把经度规范到 [-180, 180)
    """
    return (lon + 180) % 360 - 180


def make_ground_grid(lat_step=2.0, lon_step=2.0):
    """
    生成目标区域网格点
    """
    lats = np.arange(LAT_MIN, LAT_MAX + 1e-9, lat_step)
    lons = np.arange(LON_MIN, LON_MAX + 1e-9, lon_step)

    lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")

    lat_rad = deg2rad(lat_mesh.ravel())
    lon_rad = deg2rad(lon_mesh.ravel())

    return lat_rad, lon_rad, lat_mesh.shape


def satellite_subpoints(M, N, inc_deg, F, t):
    """
    计算某一时刻 t 下所有卫星的星下点经纬度。

    M: 轨道面数
    N: 每轨道面卫星数
    inc_deg: 轨道倾角，单位：度
    F: 相位因子
    t: 时间，单位：秒
    """
    inc = deg2rad(inc_deg)

    # 轨道角速度
    mean_motion = np.sqrt(MU / A**3)  # rad/s

    sat_lats = []
    sat_lons = []

    for m in range(M):
        # 第 m 个轨道面的升交点赤经，均匀分布
        raan = 2 * np.pi * m / M

        for n in range(N):
            # Walker-like 相位设置
            # 同一轨道面内均匀分布，相邻轨道面加入 F 控制的错位
            u0 = 2 * np.pi * n / N + 2 * np.pi * F * m / (M * N)

            # 当前轨道幅角
            u = u0 + mean_motion * t

            # 星下点纬度
            lat = np.arcsin(np.sin(inc) * np.sin(u))

            # 惯性系下的经度角
            lon_inertial = raan + np.arctan2(np.cos(inc) * np.sin(u), np.cos(u))

            # 转到地固系，需要减去地球自转角
            lon = lon_inertial - OMEGA_E * t

            # 经度归一化到 [-pi, pi)
            lon = (lon + np.pi) % (2 * np.pi) - np.pi

            sat_lats.append(lat)
            sat_lons.append(lon)

    return np.array(sat_lats), np.array(sat_lons)


def coverage_counts(M, N, inc_deg, F, t, ground_lats, ground_lons):
    """
    计算时刻 t 下，每个地面网格点的覆盖重数。
    """
    sat_lats, sat_lons = satellite_subpoints(M, N, inc_deg, F, t)

    # 维度：
    # ground_lats: [G]
    # sat_lats: [S]
    # 通过广播得到 [G, S]
    sin_g = np.sin(ground_lats)[:, None]
    cos_g = np.cos(ground_lats)[:, None]

    sin_s = np.sin(sat_lats)[None, :]
    cos_s = np.cos(sat_lats)[None, :]

    delta_lon = ground_lons[:, None] - sat_lons[None, :]

    cos_theta = sin_g * sin_s + cos_g * cos_s * np.cos(delta_lon)

    # 防止浮点误差
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    covered = cos_theta >= COS_COVER

    counts = covered.sum(axis=1)

    return counts


def evaluate_constellation(
    M,
    N,
    inc_deg,
    F,
    lat_step=2.0,
    lon_step=2.0,
    duration_hours=6.0,
    dt_seconds=300.0,
    verbose=True
):
    """
    验证一个星座方案是否满足单重覆盖。

    输出：
    - 是否 100% 单重覆盖
    - 覆盖率
    - 最小覆盖重数
    - 最差时刻
    - 最差点编号
    """
    ground_lats, ground_lons, grid_shape = make_ground_grid(lat_step, lon_step)

    times = np.arange(0, duration_hours * 3600 + 1e-9, dt_seconds)

    total_checks = 0
    covered_checks = 0

    global_min_count = 10**9
    worst_time = None
    worst_point_index = None

    for t in times:
        counts = coverage_counts(M, N, inc_deg, F, t, ground_lats, ground_lons)

        min_count = counts.min()

        if min_count < global_min_count:
            global_min_count = min_count
            worst_time = t
            worst_point_index = int(np.argmin(counts))

        total_checks += counts.size
        covered_checks += np.sum(counts >= 1)

        # 如果已经出现 0 覆盖，说明这个方案在当前采样下不满足
        # 这里不提前 break，是为了统计完整覆盖率
        # 如果想加速搜索，后面可以改成 break

    coverage_rate = covered_checks / total_checks

    feasible = global_min_count >= 1

    if verbose:
        print("=" * 60)
        print("Constellation Evaluation")
        print("=" * 60)
        print(f"M = {M}")
        print(f"N = {N}")
        print(f"inclination = {inc_deg} deg")
        print(f"F = {F}")
        print(f"Total satellites = {M * N}")
        print("-" * 60)
        print(f"Grid step = {lat_step} deg × {lon_step} deg")
        print(f"Time duration = {duration_hours} hours")
        print(f"Time step = {dt_seconds} s")
        print(f"Number of ground points = {len(ground_lats)}")
        print(f"Number of time samples = {len(times)}")
        print("-" * 60)
        print(f"Minimum coverage count = {global_min_count}")
        print(f"Single coverage rate = {coverage_rate * 100:.6f}%")
        print(f"Worst time = {worst_time / 3600:.3f} h")
        print(f"Worst point index = {worst_point_index}")
        print(f"Feasible = {feasible}")
        print("=" * 60)

    return {
        "feasible": feasible,
        "coverage_rate": coverage_rate,
        "min_count": global_min_count,
        "worst_time": worst_time,
        "worst_point_index": worst_point_index,
        "total_satellites": M * N
    }


# =========================
# 3. 主程序：先随便测试一个方案
# =========================

if __name__ == "__main__":
    # 这里先不是最终答案，只是测试代码能不能跑通
    result = evaluate_constellation(
        M=20,
        N=30,
        inc_deg=50,
        F=1,
        lat_step=2.0,
        lon_step=2.0,
        duration_hours=6.0,
        dt_seconds=300.0,
        verbose=True
    )