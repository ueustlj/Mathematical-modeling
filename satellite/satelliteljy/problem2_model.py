from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


# ============================================================
# 题目给定常数
# ============================================================
EARTH_RADIUS_KM = 6371.0
ORBIT_HEIGHT_KM = 550.0
ORBIT_RADIUS_KM = EARTH_RADIUS_KM + ORBIT_HEIGHT_KM
MU_KM3_S2 = 398600.4418
EARTH_ROTATION_RAD_S = 7.2921159e-5
COVERAGE_RADIUS_KM = 506.0

LATITUDE_MIN_DEG = 4.0
LATITUDE_MAX_DEG = 53.0
LONGITUDE_MIN_DEG = 73.0
LONGITUDE_MAX_DEG = 135.0

SINGLE_SATELLITE_COST_YUAN = 5_000_000.0
LAUNCH_COST_YUAN = 200_000_000.0
SATELLITES_PER_LAUNCH = 60


@dataclass(frozen=True)
class ConstellationConfig:
    """一组待评价的星座参数。"""

    planes: int
    satellites_per_plane: int
    inclination_deg: float
    phase_factor: int = 0
    raan_offset_deg: float = 0.0
    initial_phase_deg: float = 0.0

    @property
    def total_satellites(self) -> int:
        return self.planes * self.satellites_per_plane


@dataclass(frozen=True)
class SimulationResolution:
    """空间和时间离散精度。"""

    grid_step_deg: float
    time_step_s: float
    duration_s: float


@dataclass
class EvaluationResult:
    """问题二所需的主要评价指标。"""

    planes: int
    satellites_per_plane: int
    total_satellites: int
    inclination_deg: float
    phase_factor: int
    raan_offset_deg: float
    initial_phase_deg: float

    grid_step_deg: float
    time_step_s: float
    duration_h: float
    ground_points: int
    time_steps: int

    single_space_time_ratio: float
    single_full_region_time_ratio: float
    double_space_time_ratio: float
    double_full_region_time_ratio: float
    average_multiplicity: float

    worst_single_instantaneous_ratio: float
    minimum_point_single_time_ratio: float
    maximum_single_gap_minutes: float
    maximum_uncovered_points_at_one_time: int
    uncovered_time_steps: int

    single_feasible: bool
    double_95_feasible: bool

    manufacturing_cost_yuan: float
    launch_count: int
    launch_cost_yuan: float
    total_cost_yuan: float

    worst_time_minute: float
    worst_point_latitude_deg: float
    worst_point_longitude_deg: float
    maximum_gap_latitude_deg: float
    maximum_gap_longitude_deg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inclusive_axis(minimum: float, maximum: float, step: float) -> np.ndarray:
    """生成不越界且一定包含两端点的坐标轴。"""

    if step <= 0:
        raise ValueError("step 必须大于 0")
    if maximum < minimum:
        raise ValueError("maximum 必须不小于 minimum")

    values = np.arange(minimum, maximum + 1e-12, step, dtype=float)
    values = values[values <= maximum + 1e-10]

    if values.size == 0 or abs(values[0] - minimum) > 1e-10:
        values = np.insert(values, 0, minimum)

    if values[-1] < maximum - 1e-10:
        values = np.append(values, maximum)
    else:
        values[-1] = maximum

    return values


class ConstellationEvaluator:
    """
    统一的星座覆盖评价器。

    与原程序相比，主要改进：
    1. 所有搜索和验证共用同一套轨道、网格和覆盖代码；
    2. 网格严格包含题目边界且不越界；
    3. 同时评价单重和二重覆盖；
    4. 不再把 24 h 序列首尾强行相接；
    5. 使用 ECEF 固定地面网格和 KD 树邻域查询，减少重复计算。
    """

    def __init__(
        self,
        resolution: SimulationResolution,
        coverage_radius_km: float = COVERAGE_RADIUS_KM,
        coverage_margin_deg: float = 0.0,
    ) -> None:
        if resolution.grid_step_deg <= 0:
            raise ValueError("grid_step_deg 必须大于 0")
        if resolution.time_step_s <= 0:
            raise ValueError("time_step_s 必须大于 0")
        if resolution.duration_s <= 0:
            raise ValueError("duration_s 必须大于 0")

        self.resolution = resolution

        raw_coverage_angle = coverage_radius_km / EARTH_RADIUS_KM
        margin_angle = np.deg2rad(coverage_margin_deg)
        self.coverage_angle_rad = raw_coverage_angle - margin_angle

        if self.coverage_angle_rad <= 0:
            raise ValueError("覆盖安全裕度过大，导致有效覆盖角不大于 0")

        self.coverage_chord = 2.0 * np.sin(self.coverage_angle_rad / 2.0)
        self.mean_motion_rad_s = np.sqrt(MU_KM3_S2 / ORBIT_RADIUS_KM**3)

        self.latitudes = inclusive_axis(
            LATITUDE_MIN_DEG,
            LATITUDE_MAX_DEG,
            resolution.grid_step_deg,
        )
        self.longitudes = inclusive_axis(
            LONGITUDE_MIN_DEG,
            LONGITUDE_MAX_DEG,
            resolution.grid_step_deg,
        )

        longitude_grid, latitude_grid = np.meshgrid(
            self.longitudes,
            self.latitudes,
        )
        self.latitude_grid = latitude_grid
        self.longitude_grid = longitude_grid

        latitude_rad = np.deg2rad(latitude_grid.ravel())
        longitude_rad = np.deg2rad(longitude_grid.ravel())

        # 地面网格固定在 ECEF 坐标系中，只计算一次。
        self.ground_vectors_ecef = np.column_stack(
            (
                np.cos(latitude_rad) * np.cos(longitude_rad),
                np.cos(latitude_rad) * np.sin(longitude_rad),
                np.sin(latitude_rad),
            )
        )

        self.times = np.arange(
            0.0,
            resolution.duration_s,
            resolution.time_step_s,
            dtype=float,
        )

    @staticmethod
    def _validate_config(config: ConstellationConfig) -> None:
        if config.planes <= 0:
            raise ValueError("轨道面数必须大于 0")
        if config.satellites_per_plane <= 0:
            raise ValueError("每轨卫星数必须大于 0")
        if not 40.0 <= config.inclination_deg <= 60.0:
            raise ValueError("轨道倾角必须位于 40°~60°")
        if not 0 <= config.phase_factor < config.planes:
            raise ValueError("Walker 相位因子 F 必须满足 0 <= F < M")

    def _static_satellite_arrays(
        self,
        config: ConstellationConfig,
    ) -> tuple[np.ndarray, ...]:
        plane_ids = np.repeat(
            np.arange(config.planes, dtype=float),
            config.satellites_per_plane,
        )
        satellite_ids = np.tile(
            np.arange(config.satellites_per_plane, dtype=float),
            config.planes,
        )

        raan = (
            2.0 * np.pi * plane_ids / config.planes
            + np.deg2rad(config.raan_offset_deg)
        )
        phase = (
            2.0 * np.pi * satellite_ids / config.satellites_per_plane
            + 2.0
            * np.pi
            * config.phase_factor
            * plane_ids
            / (config.planes * config.satellites_per_plane)
            + np.deg2rad(config.initial_phase_deg)
        )

        inclination_rad = np.deg2rad(config.inclination_deg)

        return (
            phase,
            np.cos(raan),
            np.sin(raan),
            np.cos(inclination_rad),
            np.sin(inclination_rad),
        )

    def _satellite_vectors_ecef(
        self,
        t: float,
        static_arrays: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        phase, cos_raan, sin_raan, cos_i, sin_i = static_arrays

        argument = self.mean_motion_rad_s * t + phase
        x_orbit = np.cos(argument)
        y_orbit = np.sin(argument)

        x_inclined = x_orbit
        y_inclined = y_orbit * cos_i
        z_eci = y_orbit * sin_i

        x_eci = x_inclined * cos_raan - y_inclined * sin_raan
        y_eci = x_inclined * sin_raan + y_inclined * cos_raan

        # ECI -> ECEF：绕 z 轴旋转 -omega*t。
        earth_angle = EARTH_ROTATION_RAD_S * t
        cos_e = np.cos(earth_angle)
        sin_e = np.sin(earth_angle)

        x_ecef = cos_e * x_eci + sin_e * y_eci
        y_ecef = -sin_e * x_eci + cos_e * y_eci

        return np.column_stack((x_ecef, y_ecef, z_eci))

    def _coverage_counts(self, satellite_vectors_ecef: np.ndarray) -> np.ndarray:
        tree = cKDTree(satellite_vectors_ecef)

        # 使用单进程查询更稳定；并兼容不支持 return_length 的旧版 SciPy。
        try:
            counts = tree.query_ball_point(
                self.ground_vectors_ecef,
                r=self.coverage_chord,
                return_length=True,
            )
            return np.asarray(counts, dtype=np.int16)
        except TypeError:
            neighbours = tree.query_ball_point(
                self.ground_vectors_ecef,
                r=self.coverage_chord,
            )
            return np.fromiter(
                (len(item) for item in neighbours),
                dtype=np.int16,
                count=len(neighbours),
            )

    def evaluate(
        self,
        config: ConstellationConfig,
        return_grids: bool = False,
        progress: bool = False,
    ) -> tuple[EvaluationResult, dict[str, np.ndarray] | None]:
        self._validate_config(config)

        static_arrays = self._static_satellite_arrays(config)
        ground_number = self.ground_vectors_ecef.shape[0]
        time_number = len(self.times)

        single_covered_total = 0
        double_covered_total = 0
        multiplicity_total = 0
        single_full_time_steps = 0
        double_full_time_steps = 0
        uncovered_time_steps = 0

        point_single_covered_steps = np.zeros(ground_number, dtype=np.int32)
        point_double_covered_steps = np.zeros(ground_number, dtype=np.int32)
        current_single_gap_steps = np.zeros(ground_number, dtype=np.int32)
        maximum_single_gap_steps = np.zeros(ground_number, dtype=np.int32)

        worst_single_instantaneous_ratio = 1.0
        worst_time_index = 0
        maximum_uncovered_points = 0

        for time_index, t in enumerate(self.times):
            satellite_vectors = self._satellite_vectors_ecef(t, static_arrays)
            counts = self._coverage_counts(satellite_vectors)

            single = counts >= 1
            double = counts >= 2

            single_count = int(np.count_nonzero(single))
            double_count = int(np.count_nonzero(double))
            uncovered_count = ground_number - single_count

            single_covered_total += single_count
            double_covered_total += double_count
            multiplicity_total += int(np.sum(counts, dtype=np.int64))

            point_single_covered_steps += single
            point_double_covered_steps += double

            if single_count == ground_number:
                single_full_time_steps += 1
            else:
                uncovered_time_steps += 1

            if double_count == ground_number:
                double_full_time_steps += 1

            instantaneous_ratio = single_count / ground_number
            if instantaneous_ratio < worst_single_instantaneous_ratio:
                worst_single_instantaneous_ratio = instantaneous_ratio
                worst_time_index = time_index

            maximum_uncovered_points = max(
                maximum_uncovered_points,
                uncovered_count,
            )

            current_single_gap_steps[single] = 0
            current_single_gap_steps[~single] += 1
            maximum_single_gap_steps = np.maximum(
                maximum_single_gap_steps,
                current_single_gap_steps,
            )

            if progress and (
                time_index % max(1, time_number // 20) == 0
                or time_index == time_number - 1
            ):
                percent = 100.0 * (time_index + 1) / time_number
                print(f"仿真进度：{percent:6.2f}%")

        point_single_ratio = point_single_covered_steps / time_number
        point_double_ratio = point_double_covered_steps / time_number

        minimum_point_index = int(np.argmin(point_single_ratio))
        maximum_gap_point_index = int(np.argmax(maximum_single_gap_steps))

        grid_width = len(self.longitudes)
        minimum_point_row, minimum_point_col = divmod(
            minimum_point_index,
            grid_width,
        )
        maximum_gap_row, maximum_gap_col = divmod(
            maximum_gap_point_index,
            grid_width,
        )

        total_samples = time_number * ground_number
        single_space_time_ratio = single_covered_total / total_samples
        double_space_time_ratio = double_covered_total / total_samples
        single_full_region_time_ratio = single_full_time_steps / time_number
        double_full_region_time_ratio = double_full_time_steps / time_number
        average_multiplicity = multiplicity_total / total_samples
        maximum_single_gap_minutes = (
            int(np.max(maximum_single_gap_steps))
            * self.resolution.time_step_s
            / 60.0
        )

        # 严格单重覆盖：所有采样地点、所有采样时刻均至少有 1 颗卫星。
        single_feasible = single_covered_total == total_samples

        # 题目第（3）问采用严格解释：至少 95% 的采样时刻，全区域均为二重覆盖。
        double_95_feasible = double_full_region_time_ratio >= 0.95

        total_satellites = config.total_satellites
        manufacturing_cost = total_satellites * SINGLE_SATELLITE_COST_YUAN
        launch_count = ceil(total_satellites / SATELLITES_PER_LAUNCH)
        launch_cost = launch_count * LAUNCH_COST_YUAN

        result = EvaluationResult(
            planes=config.planes,
            satellites_per_plane=config.satellites_per_plane,
            total_satellites=total_satellites,
            inclination_deg=config.inclination_deg,
            phase_factor=config.phase_factor,
            raan_offset_deg=config.raan_offset_deg,
            initial_phase_deg=config.initial_phase_deg,
            grid_step_deg=self.resolution.grid_step_deg,
            time_step_s=self.resolution.time_step_s,
            duration_h=self.resolution.duration_s / 3600.0,
            ground_points=ground_number,
            time_steps=time_number,
            single_space_time_ratio=single_space_time_ratio,
            single_full_region_time_ratio=single_full_region_time_ratio,
            double_space_time_ratio=double_space_time_ratio,
            double_full_region_time_ratio=double_full_region_time_ratio,
            average_multiplicity=average_multiplicity,
            worst_single_instantaneous_ratio=worst_single_instantaneous_ratio,
            minimum_point_single_time_ratio=float(np.min(point_single_ratio)),
            maximum_single_gap_minutes=maximum_single_gap_minutes,
            maximum_uncovered_points_at_one_time=maximum_uncovered_points,
            uncovered_time_steps=uncovered_time_steps,
            single_feasible=single_feasible,
            double_95_feasible=double_95_feasible,
            manufacturing_cost_yuan=manufacturing_cost,
            launch_count=launch_count,
            launch_cost_yuan=launch_cost,
            total_cost_yuan=manufacturing_cost + launch_cost,
            worst_time_minute=float(self.times[worst_time_index] / 60.0),
            worst_point_latitude_deg=float(self.latitudes[minimum_point_row]),
            worst_point_longitude_deg=float(self.longitudes[minimum_point_col]),
            maximum_gap_latitude_deg=float(self.latitudes[maximum_gap_row]),
            maximum_gap_longitude_deg=float(self.longitudes[maximum_gap_col]),
        )

        grids = None
        if return_grids:
            grids = {
                "single_time_ratio_grid": point_single_ratio.reshape(
                    self.latitude_grid.shape
                ),
                "double_time_ratio_grid": point_double_ratio.reshape(
                    self.latitude_grid.shape
                ),
                "maximum_single_gap_grid_minutes": (
                    maximum_single_gap_steps.reshape(self.latitude_grid.shape)
                    * self.resolution.time_step_s
                    / 60.0
                ),
                "latitudes": self.latitudes.copy(),
                "longitudes": self.longitudes.copy(),
            }

        return result, grids


def ranking_key(result: EvaluationResult) -> tuple:
    """
    搜索阶段的排序键。

    先让可行方案排在前面；若尚无可行方案，则优先减少未完整覆盖时刻、
    提高最差时刻覆盖率、缩短最大间隙。最后才比较卫星数。
    """

    return (
        0 if result.single_feasible else 1,
        result.total_satellites if result.single_feasible else 0,
        result.uncovered_time_steps,
        -result.worst_single_instantaneous_ratio,
        -result.minimum_point_single_time_ratio,
        result.maximum_single_gap_minutes,
        -result.single_space_time_ratio,
        result.total_satellites,
    )
