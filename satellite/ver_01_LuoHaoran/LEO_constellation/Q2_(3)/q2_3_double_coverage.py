import numpy as np


# ============================================================
# 1. 星座参数
# ============================================================
# 这里暂时使用测试参数。
# 后面可以替换成问题二第（2）问得到的最优单重覆盖方案。

M_TEST = 36       # 轨道面数
N_TEST = 44       # 每个轨道面的卫星数
I_TEST = 53.0     # 轨道倾角，单位：度
F_TEST = 1        # 相邻轨道面相位参数；如果之前没有设置，先填 0


# ============================================================
# 2. 基本常数
# ============================================================

R_E = 6371e3               # 地球半径，m
H = 550e3                  # 轨道高度，m
MU = 3.986e14              # 地球引力常数，m^3/s^2
OMEGA_E = 7.292e-5         # 地球自转角速度，rad/s

# 题目给出的单颗卫星地面覆盖半径约为 506 km
GROUND_COVER_RADIUS = 506e3

# 对应的最大地心夹角
PSI_MAX = GROUND_COVER_RADIUS / R_E

# 覆盖判定中使用
COS_PSI_MAX = np.cos(PSI_MAX)


# ============================================================
# 3. 建立目标区域地面网格
# ============================================================

def create_ground_grid():
    """
    目标区域：
    纬度 4°N ~ 53°N
    经度 73°E ~ 135°E

    目前使用较粗网格进行快速测试。
    后面正式计算时再加密。
    """

    latitude_deg = np.linspace(4, 53, 26)
    longitude_deg = np.linspace(73, 135, 32)

    longitude_grid, latitude_grid = np.meshgrid(
        longitude_deg,
        latitude_deg
    )

    latitude = np.deg2rad(latitude_grid.ravel())
    longitude = np.deg2rad(longitude_grid.ravel())

    # 地面点在地固坐标系中的单位方向向量
    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)

    ground_vectors = np.column_stack((x, y, z))

    return ground_vectors


# ============================================================
# 4. 计算某一时刻所有卫星的位置
# ============================================================

def calculate_satellite_vectors(M, N, inclination_deg, F, time_s):
    """
    计算所有卫星在地固坐标系中的单位位置向量。

    参数：
    M：轨道面数
    N：每轨卫星数
    inclination_deg：轨道倾角，度
    F：轨道面间相位参数
    time_s：时间，秒
    """

    inclination = np.deg2rad(inclination_deg)

    # 每颗卫星所属的轨道面编号
    plane_index = np.repeat(np.arange(M), N)

    # 每颗卫星在轨道面内的编号
    satellite_index = np.tile(np.arange(N), M)

    # 各轨道面的升交点赤经，均匀分布
    raan = 2 * np.pi * plane_index / M

    # 卫星初始相位
    initial_phase = (
        2 * np.pi * satellite_index / N
        + 2 * np.pi * F * plane_index / (M * N)
    )

    # 圆轨道平均角速度
    orbit_radius = R_E + H

    mean_motion = np.sqrt(
        MU / orbit_radius ** 3
    )

    # 当前时刻卫星在轨道面内的角位置
    argument_of_latitude = initial_phase + mean_motion * time_s

    cos_u = np.cos(argument_of_latitude)
    sin_u = np.sin(argument_of_latitude)

    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)

    cos_i = np.cos(inclination)
    sin_i = np.sin(inclination)

    # 惯性坐标系 ECI 中的卫星单位位置向量
    x_eci = (
        cos_raan * cos_u
        - sin_raan * sin_u * cos_i
    )

    y_eci = (
        sin_raan * cos_u
        + cos_raan * sin_u * cos_i
    )

    z_eci = sin_u * sin_i

    # 考虑地球自转：ECI 转换为 ECEF
    earth_rotation_angle = OMEGA_E * time_s

    cos_theta = np.cos(earth_rotation_angle)
    sin_theta = np.sin(earth_rotation_angle)

    x_ecef = (
        cos_theta * x_eci
        + sin_theta * y_eci
    )

    y_ecef = (
        -sin_theta * x_eci
        + cos_theta * y_eci
    )

    z_ecef = z_eci

    satellite_vectors = np.column_stack(
        (x_ecef, y_ecef, z_ecef)
    )

    return satellite_vectors


# ============================================================
# 5. 评价单重覆盖与二重覆盖
# ============================================================

def evaluate_constellation(
    M,
    N,
    inclination_deg,
    F,
    simulation_hours=24,
    time_step_minutes=20
):
    """
    对给定星座进行覆盖评价。

    返回：
    1. 单重连续覆盖时间比例
    2. 严格二重覆盖时间比例
    3. 时空网格二重覆盖比例
    4. 平均覆盖重数
    5. 最低覆盖重数
    """

    ground_vectors = create_ground_grid()

    time_step_seconds = time_step_minutes * 60

    times = np.arange(
        0,
        simulation_hours * 3600 + 1,
        time_step_seconds
    )

    minimum_cover_counts = []
    double_area_ratios = []
    average_cover_counts = []

    for time_s in times:

        satellite_vectors = calculate_satellite_vectors(
            M,
            N,
            inclination_deg,
            F,
            time_s
        )

        # ground_vectors 和 satellite_vectors 都是单位向量
        # 点积等于两者地心夹角的余弦
        cosine_matrix = ground_vectors @ satellite_vectors.T

        # 判断每颗卫星是否覆盖每个地面点
        covered_matrix = cosine_matrix >= COS_PSI_MAX

        # 每个地面点当前被多少颗卫星覆盖
        cover_count = np.sum(
            covered_matrix,
            axis=1
        )

        minimum_cover_counts.append(
            np.min(cover_count)
        )

        double_area_ratios.append(
            np.mean(cover_count >= 2)
        )

        average_cover_counts.append(
            np.mean(cover_count)
        )

    minimum_cover_counts = np.array(
        minimum_cover_counts
    )

    double_area_ratios = np.array(
        double_area_ratios
    )

    average_cover_counts = np.array(
        average_cover_counts
    )

    # 每个时刻，整个区域所有点至少单重覆盖
    single_coverage_time_ratio = np.mean(
        minimum_cover_counts >= 1
    )

    # 每个时刻，整个区域所有点至少二重覆盖
    strict_double_coverage_time_ratio = np.mean(
        minimum_cover_counts >= 2
    )

    # 所有“时间-空间网格”中，达到二重覆盖的比例
    space_time_double_ratio = np.mean(
        double_area_ratios
    )

    # 整个仿真过程的平均覆盖重数
    average_multiplicity = np.mean(
        average_cover_counts
    )

    # 整个仿真过程中出现过的最低覆盖重数
    global_minimum_cover = np.min(
        minimum_cover_counts
    )

    # 最薄弱时刻
    worst_time_index = np.argmin(
        minimum_cover_counts
    )

    worst_time_hours = (
        times[worst_time_index] / 3600
    )

    return {
        "single_coverage_time_ratio":
            single_coverage_time_ratio,

        "strict_double_coverage_time_ratio":
            strict_double_coverage_time_ratio,

        "space_time_double_ratio":
            space_time_double_ratio,

        "average_multiplicity":
            average_multiplicity,

        "global_minimum_cover":
            global_minimum_cover,

        "worst_time_hours":
            worst_time_hours
    }


# ============================================================
# 6. 主程序
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("问题二第（3）问：二重覆盖评价")
    print("=" * 55)

    print(f"轨道面数 M：{M_TEST}")
    print(f"每轨卫星数 N：{N_TEST}")
    print(f"卫星总数：{M_TEST * N_TEST}")
    print(f"轨道倾角：{I_TEST:.1f}°")
    print(f"相位参数 F：{F_TEST}")
    print()

    print("正在进行 24 小时覆盖仿真，请稍候……")

    result = evaluate_constellation(
        M=M_TEST,
        N=N_TEST,
        inclination_deg=I_TEST,
        F=F_TEST,
        simulation_hours=24,
        time_step_minutes=20
    )

    print()
    print("=" * 55)
    print("仿真结果")
    print("=" * 55)

    print(
        "单重连续覆盖时间比例："
        f"{result['single_coverage_time_ratio'] * 100:.2f}%"
    )

    print(
        "严格二重覆盖时间比例："
        f"{result['strict_double_coverage_time_ratio'] * 100:.2f}%"
    )

    print(
        "时空网格二重覆盖比例："
        f"{result['space_time_double_ratio'] * 100:.2f}%"
    )

    print(
        "平均覆盖重数："
        f"{result['average_multiplicity']:.3f}"
    )

    print(
        "全程最低覆盖重数："
        f"{result['global_minimum_cover']}"
    )

    print(
        "最薄弱时刻："
        f"{result['worst_time_hours']:.2f} 小时"
    )

    print("=" * 55)

    if result["single_coverage_time_ratio"] < 1.0:
        print("结论：该方案尚不能保证单重连续覆盖。")

    elif result["strict_double_coverage_time_ratio"] >= 0.95:
        print("结论：该方案满足 95% 时间严格二重覆盖要求。")

    else:
        print("结论：该方案满足单重覆盖，但尚不满足二重覆盖要求。")