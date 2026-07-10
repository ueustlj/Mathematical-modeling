from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from problem3_common import (
    DOUBLE_COVERAGE_CONFIG,
    LIGHT_SPEED_KM_S,
    MAX_ISL_DISTANCE_KM,
    PROCESSING_DELAY_MS_PER_HOP,
    build_isl_edges,
    eci_to_ecef,
    ensure_directory,
    id_to_plane_slot,
    make_ground_grid,
    satellite_positions_eci,
    visible_satellites_and_slant_ranges,
)


# ============================================================
# 问题三第（2）问：区域内端到端最小时延路由
# ============================================================

CONFIG = DOUBLE_COVERAGE_CONFIG

# 该离散精度用于先得到可运行结果。
# 若运行时间允许，可改为 2.0° 和 300 s 进一步复核。
GROUND_GRID_STEP_DEG = 5.0
TIME_STEP_S = 600.0
DURATION_S = 24.0 * 3600.0
OUTPUT_DIR = "results_problem3_q2"


def build_augmented_routing_graph(
    satellite_positions_eci_now: np.ndarray,
    time_s: float,
    ground_unit_vectors: np.ndarray,
) -> tuple[csr_matrix, list[np.ndarray]]:
    """
    构造有向扩展图：
      1. 星间链路双向；
      2. 每个地面点设置源节点，只能从地面进入卫星；
      3. 每个地面点设置目的节点，只能从卫星离开网络到达地面。

    这种结构可一次性计算所有地面点对的最小时延，且不会把其他地面点
    错误地当作中继节点。
    """
    sat_count = CONFIG.satellite_count
    ground_count = ground_unit_vectors.shape[0]
    source_offset = sat_count
    destination_offset = sat_count + ground_count
    node_count = sat_count + 2 * ground_count

    edge_u, edge_v, edge_distance_km, _, _ = build_isl_edges(
        CONFIG,
        satellite_positions_eci_now,
        max_distance_km=MAX_ISL_DISTANCE_KM,
    )

    # 每条星间链路的权重 = 光速传播时延 + 0.5 ms/跳处理时延。
    isl_delay_ms = (
        edge_distance_km / LIGHT_SPEED_KM_S * 1000.0
        + PROCESSING_DELAY_MS_PER_HOP
    )

    rows: list[np.ndarray] = [edge_u, edge_v]
    cols: list[np.ndarray] = [edge_v, edge_u]
    data: list[np.ndarray] = [isl_delay_ms, isl_delay_ms]

    satellite_positions_ecef = eci_to_ecef(satellite_positions_eci_now, time_s)
    visible_ids, slant_ranges = visible_satellites_and_slant_ranges(
        satellite_positions_ecef,
        ground_unit_vectors,
    )

    for ground_id, (sat_ids, ranges_km) in enumerate(zip(visible_ids, slant_ranges)):
        if sat_ids.size == 0:
            continue

        access_delay_ms = ranges_km / LIGHT_SPEED_KM_S * 1000.0
        source_node = source_offset + ground_id
        destination_node = destination_offset + ground_id

        rows.append(np.full(sat_ids.size, source_node, dtype=np.int32))
        cols.append(sat_ids)
        data.append(access_delay_ms)

        rows.append(sat_ids)
        cols.append(np.full(sat_ids.size, destination_node, dtype=np.int32))
        data.append(access_delay_ms)

    row_array = np.concatenate(rows)
    col_array = np.concatenate(cols)
    data_array = np.concatenate(data)

    graph = coo_matrix(
        (data_array, (row_array, col_array)),
        shape=(node_count, node_count),
    ).tocsr()
    return graph, visible_ids


def reconstruct_path(predecessors: np.ndarray, source: int, destination: int) -> list[int]:
    """根据 scipy dijkstra 的 predecessor 数组恢复节点序列。"""
    if source == destination:
        return [source]
    if predecessors[destination] < 0:
        return []

    path = [destination]
    current = destination
    while current != source:
        current = int(predecessors[current])
        if current < 0:
            return []
        path.append(current)
    path.reverse()
    return path


def node_label(node_id: int, ground_coordinates: np.ndarray) -> str:
    sat_count = CONFIG.satellite_count
    ground_count = ground_coordinates.shape[0]

    if node_id < sat_count:
        plane, slot = id_to_plane_slot(CONFIG, node_id)
        return f"SAT(P{plane:02d},S{slot:02d})"
    if node_id < sat_count + ground_count:
        ground_id = node_id - sat_count
        lat, lon = ground_coordinates[ground_id]
        return f"SOURCE_G{ground_id}(lat={lat:.2f},lon={lon:.2f})"

    ground_id = node_id - sat_count - ground_count
    lat, lon = ground_coordinates[ground_id]
    return f"DEST_G{ground_id}(lat={lat:.2f},lon={lon:.2f})"


def main() -> None:
    output_dir = ensure_directory(OUTPUT_DIR)
    ground_coordinates, ground_unit_vectors = make_ground_grid(
        step_deg=GROUND_GRID_STEP_DEG
    )
    ground_count = ground_coordinates.shape[0]
    sat_count = CONFIG.satellite_count

    times = np.arange(0.0, DURATION_S, TIME_STEP_S)
    source_nodes = sat_count + np.arange(ground_count, dtype=np.int32)
    destination_slice = slice(sat_count + ground_count, sat_count + 2 * ground_count)
    upper_triangle = np.triu_indices(ground_count, k=1)

    snapshot_rows: list[dict] = []
    total_delay_sum = 0.0
    total_pair_count = 0
    global_max_delay = -np.inf
    global_worst_time_s = 0.0
    global_worst_source = -1
    global_worst_destination = -1

    for time_index, time_s in enumerate(times):
        positions_eci = satellite_positions_eci(CONFIG, time_s)
        graph, visible_ids = build_augmented_routing_graph(
            positions_eci,
            time_s,
            ground_unit_vectors,
        )

        all_distances = dijkstra(
            graph,
            directed=True,
            indices=source_nodes,
            return_predecessors=False,
        )
        ground_to_ground = all_distances[:, destination_slice]
        pair_delays = ground_to_ground[upper_triangle]
        finite = np.isfinite(pair_delays)
        finite_delays = pair_delays[finite]

        if finite_delays.size == 0:
            snapshot_average = float("inf")
            snapshot_maximum = float("inf")
            reachable_ratio = 0.0
        else:
            snapshot_average = float(finite_delays.mean())
            snapshot_maximum = float(finite_delays.max())
            reachable_ratio = float(finite_delays.size / pair_delays.size)
            total_delay_sum += float(finite_delays.sum())
            total_pair_count += int(finite_delays.size)

            if snapshot_maximum > global_max_delay:
                local_flat_index = int(np.argmax(np.where(finite, pair_delays, -np.inf)))
                source_ground_id = int(upper_triangle[0][local_flat_index])
                destination_ground_id = int(upper_triangle[1][local_flat_index])
                global_max_delay = snapshot_maximum
                global_worst_time_s = float(time_s)
                global_worst_source = source_ground_id
                global_worst_destination = destination_ground_id

        visible_counts = np.array([ids.size for ids in visible_ids], dtype=int)
        snapshot_rows.append(
            {
                "time_s": float(time_s),
                "time_min": float(time_s / 60.0),
                "average_end_to_end_delay_ms": snapshot_average,
                "maximum_end_to_end_delay_ms": snapshot_maximum,
                "reachable_pair_ratio": reachable_ratio,
                "minimum_visible_satellites_per_ground_point": int(visible_counts.min()),
                "mean_visible_satellites_per_ground_point": float(visible_counts.mean()),
                "maximum_visible_satellites_per_ground_point": int(visible_counts.max()),
            }
        )

        print(
            f"Q2 progress: {time_index + 1}/{len(times)}, "
            f"avg={snapshot_average:.4f} ms, max={snapshot_maximum:.4f} ms"
        )

    snapshot_summary = pd.DataFrame(snapshot_rows)
    snapshot_summary.to_csv(
        output_dir / "routing_snapshot_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_average = total_delay_sum / total_pair_count if total_pair_count else float("inf")
    meets_30ms = bool(np.isfinite(global_max_delay) and global_max_delay <= 30.0)

    with open(output_dir / "q2_conclusion.txt", "w", encoding="utf-8") as file:
        file.write(f"星座方案: M={CONFIG.planes}, N={CONFIG.satellites_per_plane}, ")
        file.write(f"i={CONFIG.inclination_deg} deg, F={CONFIG.phase_factor}\n")
        file.write(f"地面网格步长: {GROUND_GRID_STEP_DEG} deg\n")
        file.write(f"地面网格点数: {ground_count}\n")
        file.write(f"时间步长: {TIME_STEP_S} s\n")
        file.write(f"仿真时长: {DURATION_S / 3600.0} h\n")
        file.write(f"区域平均端到端时延: {overall_average:.6f} ms\n")
        file.write(f"区域最大端到端时延: {global_max_delay:.6f} ms\n")
        file.write(f"是否满足最大时延不超过 30 ms: {meets_30ms}\n")
        file.write(
            f"最差时刻: {global_worst_time_s:.1f} s "
            f"({global_worst_time_s / 60.0:.4f} min)\n"
        )
        source_lat, source_lon = ground_coordinates[global_worst_source]
        destination_lat, destination_lon = ground_coordinates[global_worst_destination]
        file.write(
            f"最差点对 A: lat={source_lat:.6f} deg, lon={source_lon:.6f} deg\n"
        )
        file.write(
            f"最差点对 B: lat={destination_lat:.6f} deg, lon={destination_lon:.6f} deg\n"
        )
        file.write(
            "路由权重: 地面接入边仅计光速传播时延；每条星间链路计光速传播时延"
            "与 0.5 ms/跳星上处理时延。\n"
        )

    # 恢复全局最差点对对应的具体路由。
    worst_positions_eci = satellite_positions_eci(CONFIG, global_worst_time_s)
    worst_graph, _ = build_augmented_routing_graph(
        worst_positions_eci,
        global_worst_time_s,
        ground_unit_vectors,
    )
    worst_source_node = sat_count + global_worst_source
    worst_destination_node = sat_count + ground_count + global_worst_destination
    worst_distances, worst_predecessors = dijkstra(
        worst_graph,
        directed=True,
        indices=worst_source_node,
        return_predecessors=True,
    )
    worst_path = reconstruct_path(
        worst_predecessors,
        worst_source_node,
        worst_destination_node,
    )

    route_rows = []
    for order, node in enumerate(worst_path):
        route_rows.append(
            {
                "order": order,
                "node_id": int(node),
                "node_label": node_label(int(node), ground_coordinates),
                "cumulative_delay_ms": float(worst_distances[node]),
            }
        )
    pd.DataFrame(route_rows).to_csv(
        output_dir / "worst_pair_route.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        snapshot_summary["time_min"],
        snapshot_summary["average_end_to_end_delay_ms"],
        label="Average delay",
    )
    plt.plot(
        snapshot_summary["time_min"],
        snapshot_summary["maximum_end_to_end_delay_ms"],
        label="Maximum delay",
    )
    plt.axhline(30.0, linestyle="--", label="30 ms requirement")
    plt.xlabel("Time (min)")
    plt.ylabel("End-to-end delay (ms)")
    plt.title("Regional end-to-end delay over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "end_to_end_delay_over_time.png", dpi=200)
    plt.close()

    print("\nQ2 finished.")
    print(f"Results saved to: {output_dir.resolve()}")
    print(f"Ground points: {ground_count}")
    print(f"Overall average delay: {overall_average:.6f} ms")
    print(f"Overall maximum delay: {global_max_delay:.6f} ms")
    print(f"Meets 30 ms requirement: {meets_30ms}")


if __name__ == "__main__":
    main()
