import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


# ============================================================
# 1. 地球与轨道基本参数
# ============================================================

EARTH_RADIUS = 6371.0              # km
ORBIT_HEIGHT = 550.0               # km
ORBIT_RADIUS = EARTH_RADIUS + ORBIT_HEIGHT

MU = 398600.4418                   # km^3 / s^2
EARTH_ROTATION_RATE = 7.2921159e-5 # rad / s


# ============================================================
# 2. 当前推荐星座参数
# ============================================================

PLANE_NUMBER = 36
SATELLITES_PER_PLANE = 44

INCLINATION = 53.0                 # degree
PHASE_FACTOR = 1

TOTAL_SATELLITES = (
    PLANE_NUMBER
    * SATELLITES_PER_PLANE
)


# ============================================================
# 3. 单星覆盖参数
# ============================================================

COVERAGE_RADIUS = 506.0            # km

# 地面弧长近似对应的地心角
COVERAGE_ANGLE = (
    COVERAGE_RADIUS
    / EARTH_RADIUS
)

# 单位球面上的等价弦长
COVERAGE_CHORD = (
    2.0
    * np.sin(COVERAGE_ANGLE / 2.0)
)


# ============================================================
# 4. 目标区域
# ============================================================

LATITUDE_MIN = 4.0
LATITUDE_MAX = 53.0

LONGITUDE_MIN = 73.0
LONGITUDE_MAX = 135.0

SIMULATION_DURATION = 24.0 * 3600.0


# ============================================================
# 5. 两级验证精度
# ============================================================

SIMULATION_CASES = [
    {
        "name": "main_resolution",
        "grid_step": 1.0,
        "time_step": 60.0
    },
    {
        "name": "sensitivity_resolution",
        "grid_step": 0.5,
        "time_step": 30.0
    }
]


# ============================================================
# 6. 输出文件夹
# ============================================================

OUTPUT_DIRECTORY = Path(
    "final_results"
)

OUTPUT_DIRECTORY.mkdir(
    exist_ok=True
)


# ============================================================
# 7. 预先建立星座编号与相位数组
# ============================================================

PLANE_IDS = np.repeat(
    np.arange(PLANE_NUMBER),
    SATELLITES_PER_PLANE
)

SATELLITE_IDS = np.tile(
    np.arange(SATELLITES_PER_PLANE),
    PLANE_NUMBER
)

# 各轨道面的升交点赤经
RAAN = (
    2.0
    * np.pi
    * PLANE_IDS
    / PLANE_NUMBER
)

# 同一个轨道面内卫星的均匀初始相位
PHASE_IN_PLANE = (
    2.0
    * np.pi
    * SATELLITE_IDS
    / SATELLITES_PER_PLANE
)

# 圆轨道平均角速度
MEAN_MOTION = np.sqrt(
    MU / ORBIT_RADIUS ** 3
)


# ============================================================
# 8. 建立目标区域网格
# ============================================================

def build_ground_grid(grid_step):
    """
    建立目标区域的经纬度网格。

    同时返回纬度面积权重 cos(latitude)，以减少普通
    经纬度网格在高纬度区域造成的面积统计偏差。
    """

    latitudes = np.arange(
        LATITUDE_MIN,
        LATITUDE_MAX + 0.5 * grid_step,
        grid_step
    )

    longitudes = np.arange(
        LONGITUDE_MIN,
        LONGITUDE_MAX + 0.5 * grid_step,
        grid_step
    )

    longitude_grid, latitude_grid = np.meshgrid(
        longitudes,
        latitudes
    )

    latitude_radians = np.deg2rad(
        latitude_grid.ravel()
    )

    longitude_radians = np.deg2rad(
        longitude_grid.ravel()
    )

    # 经纬度网格点代表的地表面积近似与 cos(latitude) 成正比
    area_weights = np.cos(
        latitude_radians
    )

    return {
        "latitudes": latitudes,
        "longitudes": longitudes,
        "latitude_grid": latitude_grid,
        "longitude_grid": longitude_grid,
        "latitude_radians": latitude_radians,
        "longitude_radians": longitude_radians,
        "area_weights": area_weights
    }


# ============================================================
# 9. 计算地面网格在惯性坐标系中的方向
# ============================================================

def calculate_ground_vectors(
        latitude_radians,
        longitude_radians,
        simulation_time
):
    """
    考虑地球自转，将地面点转换到地心惯性坐标系 ECI。
    """

    longitude_eci = (
        longitude_radians
        + EARTH_ROTATION_RATE * simulation_time
    )

    x = (
        np.cos(latitude_radians)
        * np.cos(longitude_eci)
    )

    y = (
        np.cos(latitude_radians)
        * np.sin(longitude_eci)
    )

    z = np.sin(
        latitude_radians
    )

    return np.column_stack(
        (x, y, z)
    )


# ============================================================
# 10. 计算整个星座的位置
# ============================================================

def calculate_constellation_vectors(
        simulation_time
):
    """
    计算给定时刻整个 Walker 星座在 ECI 坐标系中的单位方向向量。
    """

    plane_phase = (
        2.0
        * np.pi
        * PHASE_FACTOR
        * PLANE_IDS
        / (
            PLANE_NUMBER
            * SATELLITES_PER_PLANE
        )
    )

    argument = (
        MEAN_MOTION * simulation_time
        + PHASE_IN_PLANE
        + plane_phase
    )

    inclination_radians = np.deg2rad(
        INCLINATION
    )

    # 轨道平面中的位置
    x_orbit = np.cos(argument)
    y_orbit = np.sin(argument)

    # 绕 x 轴旋转轨道倾角
    x_inclined = x_orbit

    y_inclined = (
        y_orbit
        * np.cos(inclination_radians)
    )

    z_inclined = (
        y_orbit
        * np.sin(inclination_radians)
    )

    # 绕 z 轴旋转升交点赤经
    x_eci = (
        x_inclined * np.cos(RAAN)
        - y_inclined * np.sin(RAAN)
    )

    y_eci = (
        x_inclined * np.sin(RAAN)
        + y_inclined * np.cos(RAAN)
    )

    z_eci = z_inclined

    satellite_vectors = np.column_stack(
        (x_eci, y_eci, z_eci)
    )

    # 消除浮点误差
    satellite_vectors /= np.linalg.norm(
        satellite_vectors,
        axis=1,
        keepdims=True
    )

    return satellite_vectors


# ============================================================
# 11. 覆盖判断
# ============================================================

def calculate_coverage_counts(
        satellite_vectors,
        ground_vectors
):
    """
    使用 KD 树计算每个地面点被多少颗卫星覆盖。
    """

    tree = cKDTree(
        satellite_vectors
    )

    try:
        counts = tree.query_ball_point(
            ground_vectors,
            r=COVERAGE_CHORD,
            return_length=True
        )

        return np.asarray(
            counts,
            dtype=np.int16
        )

    except TypeError:
        neighbour_lists = tree.query_ball_point(
            ground_vectors,
            r=COVERAGE_CHORD
        )

        return np.fromiter(
            (
                len(neighbours)
                for neighbours in neighbour_lists
            ),
            dtype=np.int16,
            count=len(neighbour_lists)
        )


# ============================================================
# 12. 最大连续不满足覆盖要求的时间
# ============================================================

def calculate_maximum_gap(
        coverage_state_history,
        time_step
):
    """
    计算各地面点在仿真时间范围内，连续不满足覆盖要求的最长时间。

    coverage_state_history 为布尔数组：
    True  表示该时刻满足覆盖要求；
    False 表示该时刻不满足覆盖要求。

    注意：24 h 并不是星座相对地面的严格重复周期，
    因此这里不再把仿真末尾与开头首尾相接。
    """

    time_number, ground_number = (
        coverage_state_history.shape
    )

    unmet_history = (
        ~coverage_state_history
    )

    current_gap = np.zeros(
        ground_number,
        dtype=np.int32
    )

    maximum_gap = np.zeros(
        ground_number,
        dtype=np.int32
    )

    for time_id in range(time_number):

        current_gap = np.where(
            unmet_history[time_id],
            current_gap + 1,
            0
        )

        maximum_gap = np.maximum(
            maximum_gap,
            current_gap
        )

    maximum_gap_point_id = int(
        np.argmax(maximum_gap)
    )

    maximum_gap_minutes = (
        maximum_gap[maximum_gap_point_id]
        * time_step
        / 60.0
    )

    return {
        "gap_steps": maximum_gap,
        "maximum_gap_minutes":
            maximum_gap_minutes,
        "maximum_gap_point_id":
            maximum_gap_point_id
    }


# ============================================================
# 13. 执行一次覆盖仿真
# ============================================================

def run_simulation(
        case_name,
        grid_step,
        time_step
):
    """
    对一种空间—时间离散精度执行完整24小时仿真。

    同时计算：
    1. 单重连续覆盖指标；
    2. 二重覆盖时空平均比例；
    3. 最差地点的二重覆盖时间比例；
    4. 全区域同时二重覆盖的时间比例；
    5. 单重与二重覆盖的最大连续缺口。
    """

    grid = build_ground_grid(
        grid_step
    )

    latitude_grid = grid[
        "latitude_grid"
    ]

    longitude_grid = grid[
        "longitude_grid"
    ]

    latitude_radians = grid[
        "latitude_radians"
    ]

    longitude_radians = grid[
        "longitude_radians"
    ]

    area_weights = grid[
        "area_weights"
    ]

    area_weight_sum = np.sum(
        area_weights
    )

    times = np.arange(
        0.0,
        SIMULATION_DURATION,
        time_step
    )

    time_number = len(times)
    ground_number = latitude_grid.size

    # 单重覆盖状态：覆盖重数 >= 1
    single_coverage_history = np.zeros(
        (
            time_number,
            ground_number
        ),
        dtype=bool
    )

    # 二重覆盖状态：覆盖重数 >= 2
    double_coverage_history = np.zeros(
        (
            time_number,
            ground_number
        ),
        dtype=bool
    )

    weighted_single_coverage_sum = 0.0
    weighted_double_coverage_sum = 0.0
    weighted_multiplicity_sum = 0.0

    instantaneous_single_ratios = np.zeros(
        time_number,
        dtype=float
    )

    instantaneous_double_ratios = np.zeros(
        time_number,
        dtype=float
    )

    # 记录整个时空样本中的最低覆盖重数
    minimum_coverage_count = np.iinfo(
        np.int16
    ).max

    print("\n")
    print("=" * 70)
    print(f"开始运行：{case_name}")
    print("=" * 70)

    print(
        f"空间网格步长：{grid_step:.2f}°"
    )

    print(
        f"时间步长：{time_step:.0f} s"
    )

    print(
        f"地面网格点数：{ground_number}"
    )

    print(
        f"时间采样点数：{time_number}"
    )

    start_time = time.time()

    for time_id, simulation_time in enumerate(
            times
    ):

        satellite_vectors = (
            calculate_constellation_vectors(
                simulation_time
            )
        )

        ground_vectors = (
            calculate_ground_vectors(
                latitude_radians,
                longitude_radians,
                simulation_time
            )
        )

        coverage_counts = (
            calculate_coverage_counts(
                satellite_vectors,
                ground_vectors
            )
        )

        covered_once = (
            coverage_counts >= 1
        )

        covered_twice = (
            coverage_counts >= 2
        )

        single_coverage_history[time_id] = (
            covered_once
        )

        double_coverage_history[time_id] = (
            covered_twice
        )

        minimum_coverage_count = min(
            minimum_coverage_count,
            int(np.min(coverage_counts))
        )

        weighted_single_area = np.sum(
            area_weights
            * covered_once
        )

        weighted_double_area = np.sum(
            area_weights
            * covered_twice
        )

        weighted_single_coverage_sum += (
            weighted_single_area
        )

        weighted_double_coverage_sum += (
            weighted_double_area
        )

        weighted_multiplicity_sum += np.sum(
            area_weights
            * coverage_counts
        )

        instantaneous_single_ratios[time_id] = (
            weighted_single_area
            / area_weight_sum
        )

        instantaneous_double_ratios[time_id] = (
            weighted_double_area
            / area_weight_sum
        )

        if (
            time_id % 120 == 0
            or time_id == time_number - 1
        ):
            progress = (
                100.0
                * (time_id + 1)
                / time_number
            )

            print(
                f"仿真进度：{progress:6.2f}%"
            )

    elapsed_time = (
        time.time()
        - start_time
    )

    # ---------------- 单重覆盖指标 ----------------

    space_time_single_coverage_ratio = (
        weighted_single_coverage_sum
        / (
            time_number
            * area_weight_sum
        )
    )

    full_region_single_state = np.all(
        single_coverage_history,
        axis=1
    )

    full_region_single_coverage_ratio = np.mean(
        full_region_single_state
    )

    worst_single_time_id = int(
        np.argmin(
            instantaneous_single_ratios
        )
    )

    worst_single_instantaneous_ratio = (
        instantaneous_single_ratios[
            worst_single_time_id
        ]
    )

    worst_single_time_minutes = (
        times[worst_single_time_id]
        / 60.0
    )

    point_time_single_coverage_ratio = np.mean(
        single_coverage_history,
        axis=0
    )

    worst_single_point_id = int(
        np.argmin(
            point_time_single_coverage_ratio
        )
    )

    minimum_point_single_coverage_ratio = (
        point_time_single_coverage_ratio[
            worst_single_point_id
        ]
    )

    # ---------------- 二重覆盖指标 ----------------

    space_time_double_coverage_ratio = (
        weighted_double_coverage_sum
        / (
            time_number
            * area_weight_sum
        )
    )

    full_region_double_state = np.all(
        double_coverage_history,
        axis=1
    )

    full_region_double_coverage_ratio = np.mean(
        full_region_double_state
    )

    worst_double_time_id = int(
        np.argmin(
            instantaneous_double_ratios
        )
    )

    worst_double_instantaneous_ratio = (
        instantaneous_double_ratios[
            worst_double_time_id
        ]
    )

    worst_double_time_minutes = (
        times[worst_double_time_id]
        / 60.0
    )

    point_time_double_coverage_ratio = np.mean(
        double_coverage_history,
        axis=0
    )

    worst_double_point_id = int(
        np.argmin(
            point_time_double_coverage_ratio
        )
    )

    minimum_point_double_coverage_ratio = (
        point_time_double_coverage_ratio[
            worst_double_point_id
        ]
    )

    flat_latitudes = (
        latitude_grid.ravel()
    )

    flat_longitudes = (
        longitude_grid.ravel()
    )

    worst_single_point_latitude = (
        flat_latitudes[
            worst_single_point_id
        ]
    )

    worst_single_point_longitude = (
        flat_longitudes[
            worst_single_point_id
        ]
    )

    worst_double_point_latitude = (
        flat_latitudes[
            worst_double_point_id
        ]
    )

    worst_double_point_longitude = (
        flat_longitudes[
            worst_double_point_id
        ]
    )

    # 面积加权平均覆盖重数
    average_multiplicity = (
        weighted_multiplicity_sum
        / (
            time_number
            * area_weight_sum
        )
    )

    # 单重覆盖最大缺口
    single_gap_result = (
        calculate_maximum_gap(
            single_coverage_history,
            time_step
        )
    )

    maximum_single_gap_minutes = (
        single_gap_result[
            "maximum_gap_minutes"
        ]
    )

    maximum_single_gap_point_id = (
        single_gap_result[
            "maximum_gap_point_id"
        ]
    )

    maximum_single_gap_latitude = (
        flat_latitudes[
            maximum_single_gap_point_id
        ]
    )

    maximum_single_gap_longitude = (
        flat_longitudes[
            maximum_single_gap_point_id
        ]
    )

    # 二重覆盖最大缺口
    double_gap_result = (
        calculate_maximum_gap(
            double_coverage_history,
            time_step
        )
    )

    maximum_double_gap_minutes = (
        double_gap_result[
            "maximum_gap_minutes"
        ]
    )

    maximum_double_gap_point_id = (
        double_gap_result[
            "maximum_gap_point_id"
        ]
    )

    maximum_double_gap_latitude = (
        flat_latitudes[
            maximum_double_gap_point_id
        ]
    )

    maximum_double_gap_longitude = (
        flat_longitudes[
            maximum_double_gap_point_id
        ]
    )

    total_uncovered_cells = int(
        single_coverage_history.size
        - np.count_nonzero(
            single_coverage_history
        )
    )

    total_non_double_cells = int(
        double_coverage_history.size
        - np.count_nonzero(
            double_coverage_history
        )
    )

    maximum_uncovered_points = int(
        np.max(
            ground_number
            - np.sum(
                single_coverage_history,
                axis=1
            )
        )
    )

    maximum_non_double_points = int(
        np.max(
            ground_number
            - np.sum(
                double_coverage_history,
                axis=1
            )
        )
    )

    single_feasible = bool(
        np.all(
            single_coverage_history
        )
    )

    # 采用“最差地点至少95%的时间二重覆盖”作为主约束
    double_95_feasible = bool(
        single_feasible
        and minimum_point_double_coverage_ratio
        >= 0.95
    )

    result = {
        "case_name": case_name,

        "M": PLANE_NUMBER,
        "N": SATELLITES_PER_PLANE,
        "inclination_degree": INCLINATION,
        "phase_factor": PHASE_FACTOR,
        "total_satellites": TOTAL_SATELLITES,

        "grid_step_degree": grid_step,
        "time_step_second": time_step,
        "ground_point_number": ground_number,
        "time_sample_number": time_number,

        "minimum_coverage_count":
            minimum_coverage_count,

        "space_time_single_coverage_ratio":
            space_time_single_coverage_ratio,

        "full_region_single_coverage_ratio":
            full_region_single_coverage_ratio,

        "worst_single_instantaneous_ratio":
            worst_single_instantaneous_ratio,

        "worst_single_time_minutes":
            worst_single_time_minutes,

        "minimum_point_single_coverage_ratio":
            minimum_point_single_coverage_ratio,

        "worst_single_point_latitude":
            worst_single_point_latitude,

        "worst_single_point_longitude":
            worst_single_point_longitude,

        "space_time_double_coverage_ratio":
            space_time_double_coverage_ratio,

        "full_region_double_coverage_ratio":
            full_region_double_coverage_ratio,

        "worst_double_instantaneous_ratio":
            worst_double_instantaneous_ratio,

        "worst_double_time_minutes":
            worst_double_time_minutes,

        "minimum_point_double_coverage_ratio":
            minimum_point_double_coverage_ratio,

        "worst_double_point_latitude":
            worst_double_point_latitude,

        "worst_double_point_longitude":
            worst_double_point_longitude,

        "average_multiplicity":
            average_multiplicity,

        "maximum_single_gap_minutes":
            maximum_single_gap_minutes,

        "maximum_single_gap_latitude":
            maximum_single_gap_latitude,

        "maximum_single_gap_longitude":
            maximum_single_gap_longitude,

        "maximum_double_gap_minutes":
            maximum_double_gap_minutes,

        "maximum_double_gap_latitude":
            maximum_double_gap_latitude,

        "maximum_double_gap_longitude":
            maximum_double_gap_longitude,

        "total_uncovered_cells":
            total_uncovered_cells,

        "total_non_double_cells":
            total_non_double_cells,

        "maximum_uncovered_points":
            maximum_uncovered_points,

        "maximum_non_double_points":
            maximum_non_double_points,

        "single_feasible":
            single_feasible,

        "double_95_feasible":
            double_95_feasible,

        "elapsed_time_second":
            elapsed_time
    }

    single_coverage_grid = (
        100.0
        * point_time_single_coverage_ratio.reshape(
            latitude_grid.shape
        )
    )

    double_coverage_grid = (
        100.0
        * point_time_double_coverage_ratio.reshape(
            latitude_grid.shape
        )
    )

    save_heatmap(
        case_name,
        single_coverage_grid,
        grid_step
    )

    save_heatmap(
        f"{case_name}_double",
        double_coverage_grid,
        grid_step
    )

    print_result(
        result
    )

    return result


# ============================================================
# 14. 输出覆盖热力图
# ============================================================

def save_heatmap(
        case_name,
        coverage_grid,
        grid_step
):
    """
    保存各地面网格点24小时时间覆盖率热力图。
    """

    minimum_value = float(
        np.min(
            coverage_grid
        )
    )

    color_minimum = max(
        0.0,
        minimum_value - 0.1
    )

    plt.figure(
        figsize=(10, 7)
    )

    image = plt.imshow(
        coverage_grid,
        origin="lower",
        extent=[
            LONGITUDE_MIN,
            LONGITUDE_MAX,
            LATITUDE_MIN,
            LATITUDE_MAX
        ],
        aspect="auto",
        vmin=color_minimum,
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
        f"Regional coverage: {case_name}"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / f"{case_name}_heatmap.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


# ============================================================
# 15. 打印结果
# ============================================================

def print_result(result):
    """
    将一次仿真结果打印到终端。
    """

    print("\n")
    print("-" * 70)

    print(
        f"方案：M={result['M']}，"
        f"N={result['N']}，"
        f"i={result['inclination_degree']:.2f}°，"
        f"F={result['phase_factor']}"
    )

    print(
        f"卫星总数："
        f"{result['total_satellites']}"
    )

    print(
        "全时空最低覆盖重数："
        f"{result['minimum_coverage_count']}"
    )

    print("\n【单重覆盖】")

    print(
        "面积加权时空单重覆盖率："
        f"{100.0 * result['space_time_single_coverage_ratio']:.6f}%"
    )

    print(
        "全区域同时单重覆盖时间比例："
        f"{100.0 * result['full_region_single_coverage_ratio']:.6f}%"
    )

    print(
        "最差时刻区域单重覆盖率："
        f"{100.0 * result['worst_single_instantaneous_ratio']:.6f}%"
    )

    print(
        "最差单重覆盖时刻："
        f"{result['worst_single_time_minutes']:.2f} min"
    )

    print(
        "最差地点单重覆盖时间比例："
        f"{100.0 * result['minimum_point_single_coverage_ratio']:.6f}%"
    )

    print(
        "最差单重覆盖地点："
        f"纬度 {result['worst_single_point_latitude']:.2f}°，"
        f"经度 {result['worst_single_point_longitude']:.2f}°"
    )

    print(
        "最大单重覆盖缺口："
        f"{result['maximum_single_gap_minutes']:.2f} min"
    )

    print(
        "最大单重缺口地点："
        f"纬度 {result['maximum_single_gap_latitude']:.2f}°，"
        f"经度 {result['maximum_single_gap_longitude']:.2f}°"
    )

    print(
        "最大单时刻未单重覆盖网格点数："
        f"{result['maximum_uncovered_points']}"
    )

    print(
        "单重连续覆盖是否通过："
        f"{result['single_feasible']}"
    )

    print("\n【二重覆盖】")

    print(
        "面积加权时空二重覆盖率："
        f"{100.0 * result['space_time_double_coverage_ratio']:.6f}%"
    )

    print(
        "全区域同时二重覆盖时间比例："
        f"{100.0 * result['full_region_double_coverage_ratio']:.6f}%"
    )

    print(
        "最差时刻区域二重覆盖率："
        f"{100.0 * result['worst_double_instantaneous_ratio']:.6f}%"
    )

    print(
        "最差二重覆盖时刻："
        f"{result['worst_double_time_minutes']:.2f} min"
    )

    print(
        "最差地点二重覆盖时间比例："
        f"{100.0 * result['minimum_point_double_coverage_ratio']:.6f}%"
    )

    print(
        "最差二重覆盖地点："
        f"纬度 {result['worst_double_point_latitude']:.2f}°，"
        f"经度 {result['worst_double_point_longitude']:.2f}°"
    )

    print(
        "最大连续非二重覆盖时间："
        f"{result['maximum_double_gap_minutes']:.2f} min"
    )

    print(
        "最大非二重缺口地点："
        f"纬度 {result['maximum_double_gap_latitude']:.2f}°，"
        f"经度 {result['maximum_double_gap_longitude']:.2f}°"
    )

    print(
        "最大单时刻未达到二重覆盖网格点数："
        f"{result['maximum_non_double_points']}"
    )

    print(
        "最差地点95%时间二重覆盖是否通过："
        f"{result['double_95_feasible']}"
    )

    print("\n【综合指标】")

    print(
        "面积加权平均覆盖重数："
        f"{result['average_multiplicity']:.6f}"
    )

    print(
        "计算用时："
        f"{result['elapsed_time_second']:.2f} s"
    )

    print("-" * 70)


# ============================================================
# 16. 保存汇总结果
# ============================================================

def save_summary(results):
    """
    将两级验证结果保存至一个CSV文件。
    """

    output_path = (
        OUTPUT_DIRECTORY
        / "final_coverage_summary.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                results[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print(
        f"\n汇总结果已保存至：{output_path}"
    )


# ============================================================
# 17. 主程序
# ============================================================

def main():
    """
    依次执行主精度验证和敏感性验证。
    """

    print("=" * 70)
    print("低轨区域卫星星座覆盖性能最终模型")
    print("=" * 70)

    print(
        f"M = {PLANE_NUMBER}"
    )

    print(
        f"N = {SATELLITES_PER_PLANE}"
    )

    print(
        f"i = {INCLINATION:.2f}°"
    )

    print(
        f"F = {PHASE_FACTOR}"
    )

    print(
        f"卫星总数 = {TOTAL_SATELLITES}"
    )

    results = []

    for simulation_case in (
            SIMULATION_CASES
    ):

        result = run_simulation(
            case_name=simulation_case[
                "name"
            ],

            grid_step=simulation_case[
                "grid_step"
            ],

            time_step=simulation_case[
                "time_step"
            ]
        )

        results.append(
            result
        )

    save_summary(
        results
    )

    print("\n")
    print("=" * 70)
    print("最终解释")
    print("=" * 70)

    main_result = results[0]
    sensitivity_result = results[1]

    if main_result["feasible"]:

        print(
            "在 1° / 60 s 的主仿真精度下，"
            "该方案满足离散连续覆盖要求。"
        )

    else:

        print(
            "在 1° / 60 s 的主仿真精度下，"
            "该方案仍存在覆盖空窗。"
        )

    if sensitivity_result["feasible"]:

        print(
            "在 0.5° / 30 s 的敏感性验证中，"
            "该方案仍完全通过。"
        )

    else:

        print(
            "在 0.5° / 30 s 的敏感性验证中，"
            "发现了较小的瞬时覆盖漏洞。"
        )

        print(
            "因此该方案应描述为接近可行域边界的"
            "最小卫星规模候选方案，而不是严格全局最优解。"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()