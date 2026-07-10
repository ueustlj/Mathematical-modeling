from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components


# ============================================================
# 问题三第（2）（3）问公共基础模块
# 仅使用题目给定条件与问题二得到的 Walker 型圆轨道星座。
# ============================================================

EARTH_RADIUS_KM = 6371.0
ORBIT_HEIGHT_KM = 550.0
ORBIT_RADIUS_KM = EARTH_RADIUS_KM + ORBIT_HEIGHT_KM
MU_KM3_S2 = 3.986004418e5
EARTH_ROTATION_RAD_S = 7.2921159e-5
LIGHT_SPEED_KM_S = 299792.458
MAX_ISL_DISTANCE_KM = 5000.0
GROUND_COVERAGE_RADIUS_KM = 506.0
PROCESSING_DELAY_MS_PER_HOP = 0.5


@dataclass(frozen=True)
class ConstellationConfig:
    planes: int
    satellites_per_plane: int
    inclination_deg: float
    phase_factor: int
    raan_offset_deg: float = 0.0
    initial_phase_deg: float = 0.0
    name: str = ""

    @property
    def satellite_count(self) -> int:
        return self.planes * self.satellites_per_plane


# 问题二中的两组方案。问题三默认使用二重覆盖最终方案。
SINGLE_COVERAGE_CONFIG = ConstellationConfig(
    planes=38,
    satellites_per_plane=44,
    inclination_deg=53.0,
    phase_factor=1,
    name="single_coverage",
)

DOUBLE_COVERAGE_CONFIG = ConstellationConfig(
    planes=41,
    satellites_per_plane=48,
    inclination_deg=53.0,
    phase_factor=1,
    raan_offset_deg=0.0,
    initial_phase_deg=0.0,
    name="q3_current_model_F1",
)


def orbital_period_s() -> float:
    """550 km 圆轨道周期。"""
    return 2.0 * np.pi * np.sqrt(ORBIT_RADIUS_KM**3 / MU_KM3_S2)


def mean_motion_rad_s() -> float:
    """550 km 圆轨道平均角速度。"""
    return np.sqrt(MU_KM3_S2 / ORBIT_RADIUS_KM**3)


def satellite_positions_eci(config: ConstellationConfig, time_s: float) -> np.ndarray:
    """
    返回卫星在 ECI 坐标系中的位置，形状为 (M, N, 3)，单位 km。

    Walker 相位采用问题二中的形式：
        u_pq(0) = 2π q/N + 2π F p/(M N)
    """
    m = config.planes
    n_sat = config.satellites_per_plane

    p = np.arange(m, dtype=float)[:, None]
    q = np.arange(n_sat, dtype=float)[None, :]

    raan = np.deg2rad(config.raan_offset_deg) + 2.0 * np.pi * p / m
    u0 = (
        np.deg2rad(config.initial_phase_deg)
        + 2.0 * np.pi * q / n_sat
        + 2.0 * np.pi * config.phase_factor * p / (m * n_sat)
    )
    u = u0 + mean_motion_rad_s() * time_s

    inc = np.deg2rad(config.inclination_deg)
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    cos_i = np.cos(inc)
    sin_i = np.sin(inc)

    x = cos_raan * cos_u - sin_raan * sin_u * cos_i
    y = sin_raan * cos_u + cos_raan * sin_u * cos_i
    z = sin_u * sin_i

    return ORBIT_RADIUS_KM * np.stack((x, y, z), axis=-1)


def eci_to_ecef(positions_eci: np.ndarray, time_s: float) -> np.ndarray:
    """将 ECI 坐标转为 ECEF 坐标，单位保持为 km。"""
    theta = EARTH_ROTATION_RAD_S * time_s
    c = np.cos(theta)
    s = np.sin(theta)

    x = positions_eci[..., 0]
    y = positions_eci[..., 1]
    z = positions_eci[..., 2]

    x_ecef = c * x + s * y
    y_ecef = -s * x + c * y
    return np.stack((x_ecef, y_ecef, z), axis=-1)


def satellite_id(config: ConstellationConfig, plane: int, slot: int) -> int:
    return plane * config.satellites_per_plane + slot


def id_to_plane_slot(config: ConstellationConfig, sat_id: int) -> tuple[int, int]:
    return divmod(int(sat_id), config.satellites_per_plane)


def cyclic_shift_matching(plane_a: np.ndarray, plane_b: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """
    在两个相邻轨道面之间进行一一连接。

    由于每个轨道面内卫星等间隔排列，只需枚举 N 种循环错位：
        q -> (q + shift) mod N
    选择总距离最小的错位，既符合“最近卫星”原则，又保证每颗卫星
    在该相邻轨道面上至多连接 1 颗卫星。
    """
    n_sat = plane_a.shape[0]
    distance_matrix = np.linalg.norm(plane_a[:, None, :] - plane_b[None, :, :], axis=2)
    rows = np.arange(n_sat)

    scores = np.empty(n_sat, dtype=float)
    for shift in range(n_sat):
        cols = (rows + shift) % n_sat
        scores[shift] = distance_matrix[rows, cols].sum()

    best_shift = int(np.argmin(scores))
    matched_slots = (rows + best_shift) % n_sat
    matched_distances = distance_matrix[rows, matched_slots]
    return best_shift, matched_slots, matched_distances


def build_isl_edges(
    config: ConstellationConfig,
    positions_eci: np.ndarray,
    max_distance_km: float = MAX_ISL_DISTANCE_KM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    建立某时刻的星间链路。

    返回：
        edge_u, edge_v, edge_distance_km, edge_type, plane_pair_shifts

    edge_type:
        0 = 同轨相邻链路
        1 = 相邻轨道面链路
    """
    m = config.planes
    n_sat = config.satellites_per_plane

    edge_u: list[int] = []
    edge_v: list[int] = []
    edge_d: list[float] = []
    edge_type: list[int] = []

    # 同轨道面：每颗卫星只连接编号 +1 的相邻卫星。
    # 无向化后自然得到前、后各 1 条链路。
    for p in range(m):
        for q in range(n_sat):
            q_next = (q + 1) % n_sat
            u = satellite_id(config, p, q)
            v = satellite_id(config, p, q_next)
            d = float(np.linalg.norm(positions_eci[p, q] - positions_eci[p, q_next]))
            if d <= max_distance_km:
                edge_u.append(u)
                edge_v.append(v)
                edge_d.append(d)
                edge_type.append(0)

    # 相邻轨道面：对每一对 p 与 p+1 轨道面做循环一一匹配。
    # 每个轨道面对只处理一次，包含最后一个轨道面与第一个轨道面。
    plane_pair_shifts: list[int] = []
    for p in range(m):
        p_right = (p + 1) % m
        best_shift, matched_slots, matched_distances = cyclic_shift_matching(
            positions_eci[p], positions_eci[p_right]
        )
        plane_pair_shifts.append(best_shift)

        for q in range(n_sat):
            q_right = int(matched_slots[q])
            d = float(matched_distances[q])
            if d <= max_distance_km:
                edge_u.append(satellite_id(config, p, q))
                edge_v.append(satellite_id(config, p_right, q_right))
                edge_d.append(d)
                edge_type.append(1)

    return (
        np.asarray(edge_u, dtype=np.int32),
        np.asarray(edge_v, dtype=np.int32),
        np.asarray(edge_d, dtype=float),
        np.asarray(edge_type, dtype=np.int8),
        plane_pair_shifts,
    )


def undirected_sparse_graph(
    node_count: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_weight: np.ndarray,
) -> csr_matrix:
    """由无向边构造对称 CSR 稀疏矩阵。"""
    rows = np.concatenate((edge_u, edge_v))
    cols = np.concatenate((edge_v, edge_u))
    data = np.concatenate((edge_weight, edge_weight))
    return coo_matrix((data, (rows, cols)), shape=(node_count, node_count)).tocsr()


def topology_statistics(
    config: ConstellationConfig,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_distance_km: np.ndarray,
    edge_type: np.ndarray,
) -> dict[str, float | int | bool]:
    """计算某一时刻拓扑的基础统计量。"""
    sat_count = config.satellite_count
    graph = undirected_sparse_graph(
        sat_count,
        edge_u,
        edge_v,
        np.ones_like(edge_distance_km),
    )
    component_count, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels, minlength=component_count)
    degrees = np.asarray(graph.getnnz(axis=1)).ravel()

    intra_mask = edge_type == 0
    inter_mask = edge_type == 1

    def safe_stat(values: np.ndarray, function, default: float = np.nan) -> float:
        return float(function(values)) if values.size else default

    return {
        "satellite_count": sat_count,
        "intra_link_count": int(intra_mask.sum()),
        "inter_link_count": int(inter_mask.sum()),
        "total_link_count": int(edge_distance_km.size),
        "component_count": int(component_count),
        "largest_component_ratio": float(component_sizes.max() / sat_count),
        "connected": bool(component_count == 1),
        "min_degree": int(degrees.min()),
        "mean_degree": float(degrees.mean()),
        "max_degree": int(degrees.max(initial=0)),
        "min_link_distance_km": safe_stat(edge_distance_km, np.min),
        "mean_link_distance_km": safe_stat(edge_distance_km, np.mean),
        "max_link_distance_km": safe_stat(edge_distance_km, np.max),
        "min_inter_distance_km": safe_stat(edge_distance_km[inter_mask], np.min),
        "mean_inter_distance_km": safe_stat(edge_distance_km[inter_mask], np.mean),
        "max_inter_distance_km": safe_stat(edge_distance_km[inter_mask], np.max),
    }


def make_inclusive_axis(start: float, stop: float, step: float) -> np.ndarray:
    """生成严格包含起止边界的等步长坐标轴。"""
    if step <= 0:
        raise ValueError("step 必须为正数")
    values = np.arange(start, stop + 0.5 * step, step, dtype=float)
    values = values[values <= stop + 1e-12]
    if values.size == 0 or not np.isclose(values[-1], stop):
        values = np.append(values, stop)
    return values


def make_ground_grid(
    lat_min_deg: float = 4.0,
    lat_max_deg: float = 53.0,
    lon_min_deg: float = 73.0,
    lon_max_deg: float = 135.0,
    step_deg: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成区域地面网格。

    返回：
        coordinates_deg: (G, 2)，每行为 [lat, lon]
        unit_vectors: (G, 3)，ECEF 单位向量
    """
    lats = make_inclusive_axis(lat_min_deg, lat_max_deg, step_deg)
    lons = make_inclusive_axis(lon_min_deg, lon_max_deg, step_deg)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    lat_rad = np.deg2rad(lat_grid.ravel())
    lon_rad = np.deg2rad(lon_grid.ravel())

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    coordinates_deg = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    unit_vectors = np.column_stack((x, y, z))
    return coordinates_deg, unit_vectors


def visible_satellites_and_slant_ranges(
    satellite_positions_ecef: np.ndarray,
    ground_unit_vectors: np.ndarray,
    coverage_radius_km: float = GROUND_COVERAGE_RADIUS_KM,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    按问题二的等效地面覆盖半径判定每个地面点可见的卫星，
    并计算地面点至卫星的斜距。
    """
    sat_flat = satellite_positions_ecef.reshape(-1, 3)
    sat_unit = sat_flat / ORBIT_RADIUS_KM
    cos_threshold = np.cos(coverage_radius_km / EARTH_RADIUS_KM)

    dot_matrix = ground_unit_vectors @ sat_unit.T
    visibility = dot_matrix >= cos_threshold

    visible_ids: list[np.ndarray] = []
    slant_ranges: list[np.ndarray] = []

    ground_positions = EARTH_RADIUS_KM * ground_unit_vectors
    for g in range(ground_unit_vectors.shape[0]):
        ids = np.flatnonzero(visibility[g]).astype(np.int32)
        visible_ids.append(ids)
        if ids.size:
            ranges = np.linalg.norm(sat_flat[ids] - ground_positions[g], axis=1)
        else:
            ranges = np.empty(0, dtype=float)
        slant_ranges.append(ranges)

    return visible_ids, slant_ranges


def ensure_directory(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def estimate_dominant_period_s(values: Iterable[float], time_step_s: float) -> float:
    """
    用离散傅里叶变换估计序列的主要周期。
    仅用于辅助描述周期性，不替代解析轨道周期。
    """
    x = np.asarray(list(values), dtype=float)
    if x.size < 4 or np.allclose(x, x[0]):
        return float("nan")

    x = x - np.mean(x)
    spectrum = np.abs(np.fft.rfft(x))
    frequencies = np.fft.rfftfreq(x.size, d=time_step_s)
    spectrum[0] = 0.0

    peak = int(np.argmax(spectrum))
    if peak == 0 or frequencies[peak] <= 0:
        return float("nan")
    return float(1.0 / frequencies[peak])
