import numpy as np

from q2_3_double_coverage import (
    calculate_satellite_vectors,
    COS_PSI_MAX
)


# ============================================================
# 1. 当前候选星座参数
# ============================================================

M = 41
N = 47
INCLINATION = 53.0
F = 1


# ============================================================
# 2. 建立地面网格，并保留经纬度
# ============================================================

def create_ground_grid_with_coordinates():
    """
    目标区域：
    纬度 4°N ~ 53°N
    经度 73°E ~ 135°E

    使用与之前精细验证相同的 26 × 32 网格。
    """

    latitude_deg = np.linspace(4, 53, 26)
    longitude_deg = np.linspace(73, 135, 32)

    longitude_grid, latitude_grid = np.meshgrid(
        longitude_deg,
        latitude_deg
    )

    latitude_rad = np.deg2rad(
        latitude_grid.ravel()
    )

    longitude_rad = np.deg2rad(
        longitude_grid.ravel()
    )

    x = (
        np.cos(latitude_rad)
        * np.cos(longitude_rad)
    )

    y = (
        np.cos(latitude_rad)
        * np.sin(longitude_rad)
    )

    z = np.sin(latitude_rad)

    ground_vectors = np.column_stack(
        (x, y, z)
    )

    return (
        ground_vectors,
        latitude_grid.ravel(),
        longitude_grid.ravel()
    )


# ============================================================
# 3. 逐地点统计覆盖时间比例
# ============================================================

def evaluate_pointwise_double_coverage(
    M,
    N,
    inclination_deg,
    F,
    simulation_hours=24,
    time_step_minutes=10
):
    """
    计算每个地面网格点的：
    1. 单重覆盖时间比例
    2. 二重覆盖时间比例

    最终取所有网格点中的最小二重覆盖时间比例。
    """

    (
        ground_vectors,
        latitude_deg,
        longitude_deg
    ) = create_ground_grid_with_coordinates()

    ground_number = len(ground_vectors)

    time_step_seconds = (
        time_step_minutes * 60
    )

    # 不包含第 24 小时，避免与第 0 小时重复
    times = np.arange(
        0,
        simulation_hours * 3600,
        time_step_seconds
    )

    single_success_count = np.zeros(
        ground_number,
        dtype=int
    )

    double_success_count = np.zeros(
        ground_number,
        dtype=int
    )

    strict_single_success_times = 0
    strict_double_success_times = 0

    average_multiplicity_sum = 0.0

    for time_s in times:

        satellite_vectors = (
            calculate_satellite_vectors(
                M=M,
                N=N,
                inclination_deg=inclination_deg,
                F=F,
                time_s=time_s
            )
        )

        cosine_matrix = (
            ground_vectors
            @ satellite_vectors.T
        )

        covered_matrix = (
            cosine_matrix >= COS_PSI_MAX
        )

        cover_count = np.sum(
            covered_matrix,
            axis=1
        )

        single_mask = cover_count >= 1
        double_mask = cover_count >= 2

        single_success_count += single_mask
        double_success_count += double_mask

        if np.min(cover_count) >= 1:
            strict_single_success_times += 1

        if np.min(cover_count) >= 2:
            strict_double_success_times += 1

        average_multiplicity_sum += np.mean(
            cover_count
        )

    time_number = len(times)

    # 每个网格点各自的覆盖时间比例
    point_single_ratios = (
        single_success_count / time_number
    )

    point_double_ratios = (
        double_success_count / time_number
    )

    # 区域中最弱地点的时间覆盖率
    worst_single_ratio = np.min(
        point_single_ratios
    )

    worst_double_ratio = np.min(
        point_double_ratios
    )

    # 所有地点的平均二重覆盖时间比例
    average_double_ratio = np.mean(
        point_double_ratios
    )

    # 达到 95% 二重覆盖时间的网格点比例
    qualified_point_ratio = np.mean(
        point_double_ratios >= 0.95
    )

    strict_single_ratio = (
        strict_single_success_times
        / time_number
    )

    strict_double_ratio = (
        strict_double_success_times
        / time_number
    )

    average_multiplicity = (
        average_multiplicity_sum
        / time_number
    )

    # 找出最弱网格点
    worst_index = np.argmin(
        point_double_ratios
    )

    # 按二重覆盖率从低到高排序
    weak_indices = np.argsort(
        point_double_ratios
    )[:10]

    return {
        "worst_single_ratio":
            worst_single_ratio,

        "worst_double_ratio":
            worst_double_ratio,

        "average_double_ratio":
            average_double_ratio,

        "qualified_point_ratio":
            qualified_point_ratio,

        "strict_single_ratio":
            strict_single_ratio,

        "strict_double_ratio":
            strict_double_ratio,

        "average_multiplicity":
            average_multiplicity,

        "worst_latitude":
            latitude_deg[worst_index],

        "worst_longitude":
            longitude_deg[worst_index],

        "weak_indices":
            weak_indices,

        "latitude_deg":
            latitude_deg,

        "longitude_deg":
            longitude_deg,

        "point_double_ratios":
            point_double_ratios
    }


# ============================================================
# 4. 主程序
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("问题二第（3）问：逐地点二重覆盖时间比例验证")
    print("=" * 70)

    print(f"轨道面数 M：{M}")
    print(f"每轨卫星数 N：{N}")
    print(f"卫星总数：{M * N}")
    print(f"轨道倾角：{INCLINATION:.1f}°")
    print(f"相位参数 F：{F}")
    print()

    print("正在进行 24 小时逐地点覆盖统计……")
    print("地面网格：26 × 32")
    print("时间步长：10 分钟")
    print()

    result = evaluate_pointwise_double_coverage(
        M=M,
        N=N,
        inclination_deg=INCLINATION,
        F=F,
        simulation_hours=24,
        time_step_minutes=10
    )

    print()
    print("=" * 70)
    print("逐地点评价结果")
    print("=" * 70)

    print(
        "最弱地点单重覆盖时间比例："
        f"{result['worst_single_ratio'] * 100:.2f}%"
    )

    print(
        "最弱地点二重覆盖时间比例："
        f"{result['worst_double_ratio'] * 100:.2f}%"
    )

    print(
        "区域平均二重覆盖时间比例："
        f"{result['average_double_ratio'] * 100:.2f}%"
    )

    print(
        "达到 95% 二重覆盖时间的网格点比例："
        f"{result['qualified_point_ratio'] * 100:.2f}%"
    )

    print(
        "全区域同时单重覆盖时间比例："
        f"{result['strict_single_ratio'] * 100:.2f}%"
    )

    print(
        "全区域同时二重覆盖时间比例："
        f"{result['strict_double_ratio'] * 100:.2f}%"
    )

    print(
        "平均覆盖重数："
        f"{result['average_multiplicity']:.3f}"
    )

    print(
        "最弱网格点位置："
        f"纬度 {result['worst_latitude']:.2f}°N，"
        f"经度 {result['worst_longitude']:.2f}°E"
    )

    print()
    print("-" * 70)
    print("二重覆盖时间比例最低的 10 个网格点")
    print("-" * 70)

    for rank, index in enumerate(
        result["weak_indices"],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"纬度="
            f"{result['latitude_deg'][index]:6.2f}°N，"
            f"经度="
            f"{result['longitude_deg'][index]:7.2f}°E，"
            f"二重覆盖率="
            f"{result['point_double_ratios'][index] * 100:6.2f}%"
        )

    print()
    print("=" * 70)

    if (
        result["worst_single_ratio"] >= 0.999999
        and
        result["worst_double_ratio"] >= 0.95
    ):

        print(
            "结论：目标区域内每一个网格点的"
            "二重覆盖时间比例均不低于 95%，方案通过。"
        )

    elif result["worst_single_ratio"] >= 0.999999:

        print(
            "结论：能够保持单重连续覆盖，"
            "但最弱地点的二重覆盖时间比例低于 95%。"
        )

    else:

        print(
            "结论：部分地点尚不能保持单重连续覆盖。"
        )