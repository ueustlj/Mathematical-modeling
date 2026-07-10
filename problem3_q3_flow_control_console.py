from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from problem3_base import (
    DOUBLE_COVERAGE_CONFIG,
    LIGHT_SPEED_KM_S,
    MAX_ISL_DISTANCE_KM,
    PROCESSING_DELAY_MS_PER_HOP,
    build_isl_edges,
    eci_to_ecef,
    ensure_directory,
    make_ground_grid,
    satellite_positions_eci,
    undirected_sparse_graph,
    visible_satellites_and_slant_ranges,
)

# ============================================================
# 问题三第（3）问：流量分配与拥塞控制
# ============================================================

CONFIG = DOUBLE_COVERAGE_CONFIG

# 题目给定参数。
ACCESS_CAPACITY_GBPS = 20.0
PEAK_TO_AVERAGE_RATIO = 1.5

# 题目未给出，代码中作统一假设，论文中需明确说明。
ISL_CAPACITY_GBPS = 20.0
AVERAGE_TOTAL_TRAFFIC_GBPS = 300.0
AVERAGE_PACKET_MBIT = 1.0

# 为防止链路工作在理论饱和点，容量约束保留 5% 拥塞裕度。
UTILIZATION_LIMIT = 0.95

# 业务离散与候选路由参数。
GROUND_GRID_STEP_DEG = 12.0
ACCESS_CANDIDATES_PER_GROUND = 2
K_PATHS_PER_PAIR = 3
TIME_SNAPSHOT_HOURS = np.arange(0.0, 24.0, 3.0)
OUTPUT_DIR = "results_q3_3_flow_control"


@dataclass
class CandidatePath:
    commodity_id: int
    source_ground: int
    destination_ground: int
    source_satellite: int
    destination_satellite: int
    satellite_nodes: list[int]
    isl_edges: list[tuple[int, int]]
    base_delay_ms: float


def canonical_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def reconstruct_satellite_path(
    predecessor_row: np.ndarray,
    source: int,
    destination: int,
) -> list[int]:
    if source == destination:
        return [source]
    if predecessor_row[destination] < 0:
        return []

    path = [destination]
    current = destination
    while current != source:
        current = int(predecessor_row[current])
        if current < 0:
            return []
        path.append(current)
    path.reverse()
    return path


def traffic_total_gbps(time_hour: float) -> float:
    """
    日周期业务量：日均值为 AVERAGE_TOTAL_TRAFFIC_GBPS，
    峰值恰为平均值的 1.5 倍，谷值为平均值的 0.5 倍。
    """
    factor = 1.0 + 0.5 * np.sin(2.0 * np.pi * time_hour / 24.0)
    return float(AVERAGE_TOTAL_TRAFFIC_GBPS * factor)


def select_access_candidates(
    visible_ids: list[np.ndarray],
    slant_ranges: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    candidate_ids: list[np.ndarray] = []
    candidate_ranges: list[np.ndarray] = []

    for ids, ranges in zip(visible_ids, slant_ranges):
        if ids.size == 0:
            candidate_ids.append(np.empty(0, dtype=np.int32))
            candidate_ranges.append(np.empty(0, dtype=float))
            continue

        order = np.argsort(ranges)[:ACCESS_CANDIDATES_PER_GROUND]
        candidate_ids.append(ids[order].astype(np.int32))
        candidate_ranges.append(ranges[order].astype(float))

    return candidate_ids, candidate_ranges


def build_candidate_paths(
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_distance_km: np.ndarray,
    access_ids: list[np.ndarray],
    access_ranges: list[np.ndarray],
) -> tuple[list[CandidatePath], list[tuple[int, int]], np.ndarray]:
    sat_count = CONFIG.satellite_count
    edge_delay_ms = (
        edge_distance_km / LIGHT_SPEED_KM_S * 1000.0
        + PROCESSING_DELAY_MS_PER_HOP
    )
    graph = undirected_sparse_graph(sat_count, edge_u, edge_v, edge_delay_ms)

    physical_edges = [canonical_edge(int(u), int(v)) for u, v in zip(edge_u, edge_v)]
    edge_delay_lookup = {
        canonical_edge(int(u), int(v)): float(delay)
        for u, v, delay in zip(edge_u, edge_v, edge_delay_ms)
    }

    unique_access_sats = sorted(
        {int(sat) for ids in access_ids for sat in ids.tolist()}
    )
    if not unique_access_sats:
        return [], physical_edges, edge_delay_ms

    distance_matrix, predecessor_matrix = dijkstra(
        graph,
        directed=False,
        indices=np.asarray(unique_access_sats, dtype=np.int32),
        return_predecessors=True,
    )
    source_row = {sat: idx for idx, sat in enumerate(unique_access_sats)}

    ground_count = len(access_ids)
    candidates: list[CandidatePath] = []
    commodity_id = 0

    for source_ground in range(ground_count):
        for destination_ground in range(source_ground + 1, ground_count):
            path_options: list[CandidatePath] = []

            for source_local, source_sat in enumerate(access_ids[source_ground]):
                source_sat = int(source_sat)
                source_access_delay = (
                    float(access_ranges[source_ground][source_local])
                    / LIGHT_SPEED_KM_S
                    * 1000.0
                )
                row = source_row[source_sat]

                for destination_local, destination_sat in enumerate(
                    access_ids[destination_ground]
                ):
                    destination_sat = int(destination_sat)
                    destination_access_delay = (
                        float(access_ranges[destination_ground][destination_local])
                        / LIGHT_SPEED_KM_S
                        * 1000.0
                    )

                    if not np.isfinite(distance_matrix[row, destination_sat]):
                        continue

                    sat_path = reconstruct_satellite_path(
                        predecessor_matrix[row],
                        source_sat,
                        destination_sat,
                    )
                    if not sat_path:
                        continue

                    isl_edges = [
                        canonical_edge(sat_path[i], sat_path[i + 1])
                        for i in range(len(sat_path) - 1)
                    ]
                    isl_delay = sum(edge_delay_lookup[e] for e in isl_edges)
                    base_delay = source_access_delay + isl_delay + destination_access_delay

                    path_options.append(
                        CandidatePath(
                            commodity_id=commodity_id,
                            source_ground=source_ground,
                            destination_ground=destination_ground,
                            source_satellite=source_sat,
                            destination_satellite=destination_sat,
                            satellite_nodes=sat_path,
                            isl_edges=isl_edges,
                            base_delay_ms=float(base_delay),
                        )
                    )

            # 去重并保留最低时延的 K 条路径。
            unique: dict[tuple[int, int, tuple[tuple[int, int], ...]], CandidatePath] = {}
            for path in path_options:
                key = (
                    path.source_satellite,
                    path.destination_satellite,
                    tuple(path.isl_edges),
                )
                if key not in unique or path.base_delay_ms < unique[key].base_delay_ms:
                    unique[key] = path

            selected = sorted(unique.values(), key=lambda item: item.base_delay_ms)[
                :K_PATHS_PER_PAIR
            ]
            candidates.extend(selected)
            commodity_id += 1

    return candidates, physical_edges, edge_delay_ms


def solve_fair_flow_allocation(
    candidates: list[CandidatePath],
    commodity_count: int,
    pair_demand_gbps: float,
    physical_edges: list[tuple[int, int]],
) -> tuple[np.ndarray, float, dict[tuple[int, int], float], dict[int, float]]:
    """
    两阶段线性规划：
    1. 最大化所有业务点对共同满足比例 alpha；
    2. 固定 alpha 后，最小化总基础时延。
    """
    path_count = len(candidates)
    alpha_index = path_count
    variable_count = path_count + 1

    paths_by_commodity: list[list[int]] = [[] for _ in range(commodity_count)]
    for path_index, path in enumerate(candidates):
        paths_by_commodity[path.commodity_id].append(path_index)

    # 等式：每个点对的路径流量之和 = alpha * 需求。
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    b_eq = np.zeros(commodity_count, dtype=float)

    for commodity_id, indices in enumerate(paths_by_commodity):
        for path_index in indices:
            eq_rows.append(commodity_id)
            eq_cols.append(path_index)
            eq_data.append(1.0)
        eq_rows.append(commodity_id)
        eq_cols.append(alpha_index)
        eq_data.append(-pair_demand_gbps)

    A_eq = coo_matrix(
        (eq_data, (eq_rows, eq_cols)),
        shape=(commodity_count, variable_count),
    ).tocsr()

    # 不等式：星间链路容量与单星接入容量。
    edge_to_row = {edge: row for row, edge in enumerate(physical_edges)}
    access_sats = sorted(
        {
            path.source_satellite
            for path in candidates
        }
        | {
            path.destination_satellite
            for path in candidates
        }
    )
    access_to_row = {
        sat: len(physical_edges) + idx for idx, sat in enumerate(access_sats)
    }
    capacity_row_count = len(physical_edges) + len(access_sats)

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []

    for path_index, path in enumerate(candidates):
        for edge in path.isl_edges:
            ub_rows.append(edge_to_row[edge])
            ub_cols.append(path_index)
            ub_data.append(1.0)

        ub_rows.append(access_to_row[path.source_satellite])
        ub_cols.append(path_index)
        ub_data.append(1.0)

        ub_rows.append(access_to_row[path.destination_satellite])
        ub_cols.append(path_index)
        ub_data.append(1.0)

    A_ub = coo_matrix(
        (ub_data, (ub_rows, ub_cols)),
        shape=(capacity_row_count, variable_count),
    ).tocsr()
    b_ub = np.concatenate(
        (
            np.full(
                len(physical_edges),
                ISL_CAPACITY_GBPS * UTILIZATION_LIMIT,
                dtype=float,
            ),
            np.full(
                len(access_sats),
                ACCESS_CAPACITY_GBPS * UTILIZATION_LIMIT,
                dtype=float,
            ),
        )
    )

    bounds = [(0.0, None)] * path_count + [(0.0, 1.0)]

    # 第一阶段：最大化 alpha。
    objective_stage1 = np.zeros(variable_count, dtype=float)
    objective_stage1[alpha_index] = -1.0
    result_stage1 = linprog(
        objective_stage1,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result_stage1.success:
        raise RuntimeError(f"第一阶段线性规划失败: {result_stage1.message}")

    alpha_star = float(result_stage1.x[alpha_index])

    # 第二阶段：在最大吞吐率下最小化总基础时延。
    objective_stage2 = np.zeros(variable_count, dtype=float)
    objective_stage2[:path_count] = np.array(
        [path.base_delay_ms for path in candidates],
        dtype=float,
    )
    bounds_stage2 = [(0.0, None)] * path_count + [(alpha_star, alpha_star)]
    result_stage2 = linprog(
        objective_stage2,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds_stage2,
        method="highs",
    )
    if not result_stage2.success:
        raise RuntimeError(f"第二阶段线性规划失败: {result_stage2.message}")

    path_flows = result_stage2.x[:path_count]

    edge_loads = {edge: 0.0 for edge in physical_edges}
    access_loads = {sat: 0.0 for sat in access_sats}
    for flow, path in zip(path_flows, candidates):
        if flow <= 1e-12:
            continue
        for edge in path.isl_edges:
            edge_loads[edge] += float(flow)
        access_loads[path.source_satellite] += float(flow)
        access_loads[path.destination_satellite] += float(flow)

    return path_flows, alpha_star, edge_loads, access_loads


def evaluate_delay_with_congestion(
    candidates: list[CandidatePath],
    path_flows: np.ndarray,
    edge_loads: dict[tuple[int, int], float],
    access_loads: dict[int, float],
) -> tuple[float, float]:
    packet_gbit = AVERAGE_PACKET_MBIT / 1000.0

    edge_queue_ms = {}
    for edge, load in edge_loads.items():
        residual = max(ISL_CAPACITY_GBPS - load, 1e-9)
        edge_queue_ms[edge] = packet_gbit / residual * 1000.0

    access_queue_ms = {}
    for sat, load in access_loads.items():
        residual = max(ACCESS_CAPACITY_GBPS - load, 1e-9)
        access_queue_ms[sat] = packet_gbit / residual * 1000.0

    total_flow = float(path_flows.sum())
    if total_flow <= 0.0:
        return float("inf"), float("inf")

    weighted_delay_sum = 0.0
    maximum_used_path_delay = 0.0
    for flow, path in zip(path_flows, candidates):
        if flow <= 1e-12:
            continue
        delay = path.base_delay_ms
        delay += sum(edge_queue_ms[edge] for edge in path.isl_edges)
        delay += access_queue_ms[path.source_satellite]
        delay += access_queue_ms[path.destination_satellite]
        weighted_delay_sum += float(flow) * delay
        maximum_used_path_delay = max(maximum_used_path_delay, delay)

    return weighted_delay_sum / total_flow, maximum_used_path_delay


def save_resource_loads(
    output_dir: Path,
    time_hour: float,
    edge_loads: dict[tuple[int, int], float],
    access_loads: dict[int, float],
) -> None:
    edge_rows = [
        {
            "time_hour": time_hour,
            "satellite_u": edge[0],
            "satellite_v": edge[1],
            "load_gbps": load,
            "capacity_gbps": ISL_CAPACITY_GBPS,
            "utilization": load / ISL_CAPACITY_GBPS,
        }
        for edge, load in edge_loads.items()
    ]
    pd.DataFrame(edge_rows).to_csv(
        output_dir / f"isl_loads_{time_hour:05.1f}h.csv",
        index=False,
        encoding="utf-8-sig",
    )

    access_rows = [
        {
            "time_hour": time_hour,
            "satellite": sat,
            "load_gbps": load,
            "capacity_gbps": ACCESS_CAPACITY_GBPS,
            "utilization": load / ACCESS_CAPACITY_GBPS,
        }
        for sat, load in access_loads.items()
    ]
    pd.DataFrame(access_rows).to_csv(
        output_dir / f"access_loads_{time_hour:05.1f}h.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    output_dir = ensure_directory(OUTPUT_DIR)
    ground_coordinates, ground_unit_vectors = make_ground_grid(
        step_deg=GROUND_GRID_STEP_DEG
    )
    ground_count = ground_coordinates.shape[0]
    commodity_count = ground_count * (ground_count - 1) // 2

    summary_rows: list[dict[str, float | int]] = []

    for snapshot_index, time_hour in enumerate(TIME_SNAPSHOT_HOURS):
        time_s = float(time_hour * 3600.0)
        total_demand = traffic_total_gbps(float(time_hour))
        pair_demand = total_demand / commodity_count

        positions_eci = satellite_positions_eci(CONFIG, time_s)
        edge_u, edge_v, edge_distance_km, _, _ = build_isl_edges(
            CONFIG,
            positions_eci,
            max_distance_km=MAX_ISL_DISTANCE_KM,
        )

        positions_ecef = eci_to_ecef(positions_eci, time_s)
        visible_ids, slant_ranges = visible_satellites_and_slant_ranges(
            positions_ecef,
            ground_unit_vectors,
        )
        access_ids, access_ranges = select_access_candidates(
            visible_ids,
            slant_ranges,
        )

        if any(ids.size == 0 for ids in access_ids):
            raise RuntimeError(
                f"t={time_hour:.1f} h 时存在无可见卫星的地面点，请检查星座或网格。"
            )

        candidates, physical_edges, _ = build_candidate_paths(
            edge_u,
            edge_v,
            edge_distance_km,
            access_ids,
            access_ranges,
        )

        expected_commodity_ids = {path.commodity_id for path in candidates}
        if len(expected_commodity_ids) != commodity_count:
            raise RuntimeError(
                f"t={time_hour:.1f} h 时仅有 {len(expected_commodity_ids)}/"
                f"{commodity_count} 个业务点对获得候选路径。"
            )

        path_flows, alpha, edge_loads, access_loads = solve_fair_flow_allocation(
            candidates,
            commodity_count,
            pair_demand,
            physical_edges,
        )

        average_delay, maximum_delay = evaluate_delay_with_congestion(
            candidates,
            path_flows,
            edge_loads,
            access_loads,
        )

        accepted_throughput = alpha * total_demand
        max_isl_utilization = max(edge_loads.values(), default=0.0) / ISL_CAPACITY_GBPS
        max_access_utilization = (
            max(access_loads.values(), default=0.0) / ACCESS_CAPACITY_GBPS
        )

        summary_rows.append(
            {
                "time_hour": float(time_hour),
                "offered_traffic_gbps": total_demand,
                "accepted_throughput_gbps": accepted_throughput,
                "throughput_ratio": alpha,
                "average_delay_ms": average_delay,
                "maximum_used_path_delay_ms": maximum_delay,
                "maximum_isl_utilization": max_isl_utilization,
                "maximum_access_utilization": max_access_utilization,
                "candidate_path_count": len(candidates),
            }
        )

        save_resource_loads(
            output_dir,
            float(time_hour),
            edge_loads,
            access_loads,
        )

        print(
            f"Q3(3) progress {snapshot_index + 1}/{len(TIME_SNAPSHOT_HOURS)}: "
            f"t={time_hour:.1f} h, demand={total_demand:.2f} Gbps, "
            f"throughput={accepted_throughput:.2f} Gbps, "
            f"ratio={alpha:.4f}, avg delay={average_delay:.3f} ms, "
            f"max path delay={maximum_delay:.3f} ms, "
            f"max ISL util={max_isl_utilization * 100:.2f}%, "
            f"max access util={max_access_utilization * 100:.2f}%"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / "flow_performance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    peak_row = summary.loc[summary["offered_traffic_gbps"].idxmax()]
    max_delay_row = summary.loc[summary["maximum_used_path_delay_ms"].idxmax()]

    # 两种“平均吞吐率”含义必须区分：
    # 1) snapshot_mean_ratio：8 个时刻吞吐率的算术平均；
    # 2) daily_weighted_ratio：全天接纳业务总量 / 全天需求总量。
    # 论文评价系统总吞吐能力时，建议采用第 2 个。
    snapshot_mean_ratio = float(summary["throughput_ratio"].mean())
    daily_weighted_ratio = float(
        summary["accepted_throughput_gbps"].sum()
        / summary["offered_traffic_gbps"].sum()
    )
    mean_offered_traffic = float(summary["offered_traffic_gbps"].mean())
    mean_accepted_throughput = float(
        summary["accepted_throughput_gbps"].mean()
    )

    # 时延按实际接纳吞吐量加权，更符合全日业务体验。
    throughput_weighted_delay = float(
        (
            summary["average_delay_ms"]
            * summary["accepted_throughput_gbps"]
        ).sum()
        / summary["accepted_throughput_gbps"].sum()
    )

    with open(output_dir / "q3_3_conclusion.txt", "w", encoding="utf-8") as file:
        file.write(
            f"星座参数: M={CONFIG.planes}, N={CONFIG.satellites_per_plane}, "
            f"i={CONFIG.inclination_deg} deg, F={CONFIG.phase_factor}\n"
        )
        file.write(f"地面业务网格步长: {GROUND_GRID_STEP_DEG} deg\n")
        file.write(f"地面业务点数: {ground_count}\n")
        file.write(f"平均总业务量假设: {AVERAGE_TOTAL_TRAFFIC_GBPS:.3f} Gbps\n")
        file.write(
            f"峰值总业务量: {AVERAGE_TOTAL_TRAFFIC_GBPS * PEAK_TO_AVERAGE_RATIO:.3f} Gbps\n"
        )
        file.write(f"单星接入容量: {ACCESS_CAPACITY_GBPS:.3f} Gbps\n")
        file.write(f"单条星间链路容量假设: {ISL_CAPACITY_GBPS:.3f} Gbps\n")
        file.write(f"容量使用上限: {UTILIZATION_LIMIT * 100:.1f}%\n")
        file.write(
            f"8 个采样时刻吞吐率算术平均: "
            f"{snapshot_mean_ratio * 100:.4f}%\n"
        )
        file.write(
            f"全日加权吞吐率（接纳总量/需求总量）: "
            f"{daily_weighted_ratio * 100:.4f}%\n"
        )
        file.write(f"平均需求流量: {mean_offered_traffic:.6f} Gbps\n")
        file.write(
            f"平均接纳吞吐量: {mean_accepted_throughput:.6f} Gbps\n"
        )
        file.write(
            f"全日吞吐量加权平均时延: "
            f"{throughput_weighted_delay:.6f} ms\n"
        )
        file.write(
            f"峰值时刻: {peak_row['time_hour']:.1f} h，"
            f"需求 {peak_row['offered_traffic_gbps']:.3f} Gbps，"
            f"实际吞吐量 {peak_row['accepted_throughput_gbps']:.3f} Gbps，"
            f"吞吐率 {peak_row['throughput_ratio'] * 100:.4f}%\n"
        )
        file.write(
            f"峰值时刻平均时延: {peak_row['average_delay_ms']:.6f} ms\n"
        )
        file.write(
            f"峰值时刻最大已用路径时延: "
            f"{peak_row['maximum_used_path_delay_ms']:.6f} ms\n"
        )
        file.write(
            f"全日最大已用路径时延: "
            f"{max_delay_row['maximum_used_path_delay_ms']:.6f} ms，"
            f"发生于 {max_delay_row['time_hour']:.1f} h\n"
        )
        file.write(
            "优化策略: 对每个业务点对保留多条候选路径，先最大化所有点对的共同"
            "满足比例，再在该吞吐率下最小化总基础时延，并将链路利用率限制在 95%。\n"
        )

    plt.figure(figsize=(9, 5))
    plt.plot(summary["time_hour"], summary["offered_traffic_gbps"], marker="o", label="Demand")
    plt.plot(summary["time_hour"], summary["accepted_throughput_gbps"], marker="o", label="Throughput")
    plt.xlabel("Time / h")
    plt.ylabel("Traffic / Gbps")
    plt.title("Offered traffic and accepted throughput")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "traffic_and_throughput.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(summary["time_hour"], summary["average_delay_ms"], marker="o", label="Average delay")
    plt.plot(
        summary["time_hour"],
        summary["maximum_used_path_delay_ms"],
        marker="o",
        label="Maximum used-path delay",
    )
    plt.xlabel("Time / h")
    plt.ylabel("Delay / ms")
    plt.title("Delay under time-varying traffic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "delay_performance.png", dpi=200)
    plt.close()

    print("\n" + "=" * 68)
    print("Q3(3) 最终汇总")
    print("=" * 68)
    print(f"星座参数             : M={CONFIG.planes}, "
          f"N={CONFIG.satellites_per_plane}, "
          f"i={CONFIG.inclination_deg:.1f} deg, "
          f"F={CONFIG.phase_factor}")
    print(f"地面业务点数         : {ground_count}")
    print(f"平均需求流量         : {mean_offered_traffic:.3f} Gbps")
    print(f"平均接纳吞吐量       : {mean_accepted_throughput:.3f} Gbps")
    print(
        f"采样时刻吞吐率均值   : {snapshot_mean_ratio * 100:.4f}% "
        "（仅为 8 个比例的算术平均）"
    )
    print(
        f"全日加权吞吐率       : {daily_weighted_ratio * 100:.4f}% "
        "（建议写入论文）"
    )
    print(f"吞吐量加权平均时延   : {throughput_weighted_delay:.6f} ms")
    print(
        f"峰值需求时刻         : {peak_row['time_hour']:.1f} h, "
        f"需求={peak_row['offered_traffic_gbps']:.3f} Gbps, "
        f"吞吐量={peak_row['accepted_throughput_gbps']:.3f} Gbps, "
        f"吞吐率={peak_row['throughput_ratio'] * 100:.4f}%"
    )
    print(
        f"峰值时刻平均时延     : {peak_row['average_delay_ms']:.6f} ms"
    )
    print(
        f"峰值时刻最大路径时延 : "
        f"{peak_row['maximum_used_path_delay_ms']:.6f} ms"
    )
    print(
        f"全日最大路径时延     : "
        f"{max_delay_row['maximum_used_path_delay_ms']:.6f} ms "
        f"(t={max_delay_row['time_hour']:.1f} h)"
    )
    print(
        f"峰值时刻最大 ISL 利用率: "
        f"{peak_row['maximum_isl_utilization'] * 100:.2f}%"
    )
    print(
        f"峰值时刻最大接入利用率: "
        f"{peak_row['maximum_access_utilization'] * 100:.2f}%"
    )
    print("=" * 68)
    print("Q3(3) finished.")
    print(f"Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
