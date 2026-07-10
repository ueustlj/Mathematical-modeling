from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from problem3_common import (
    DOUBLE_COVERAGE_CONFIG,
    MAX_ISL_DISTANCE_KM,
    build_isl_edges,
    cyclic_shift_matching,
    ensure_directory,
    estimate_dominant_period_s,
    orbital_period_s,
    satellite_positions_eci,
    topology_statistics,
)


# ============================================================
# 问题三第（1）问：星间链路时变拓扑
# ============================================================

CONFIG = DOUBLE_COVERAGE_CONFIG
TIME_STEP_S = 30.0
SIMULATION_ORBITS = 2.0
OUTPUT_DIR = "results_problem3_q1"


def main() -> None:
    output_dir = ensure_directory(OUTPUT_DIR)
    period_s = orbital_period_s()
    duration_s = SIMULATION_ORBITS * period_s
    times = np.arange(0.0, duration_s + 0.5 * TIME_STEP_S, TIME_STEP_S)

    # 固定跟踪一对不同轨道面的卫星，用于显示距离随时间的周期变化。
    initial_positions = satellite_positions_eci(CONFIG, 0.0)
    _, initial_matches, _ = cyclic_shift_matching(initial_positions[0], initial_positions[1])
    tracked_plane_0_slot = 0
    tracked_plane_1_slot = int(initial_matches[tracked_plane_0_slot])

    summary_rows: list[dict] = []
    tracked_rows: list[dict] = []

    for index, time_s in enumerate(times):
        positions = satellite_positions_eci(CONFIG, time_s)
        edge_u, edge_v, edge_d, edge_type, shifts = build_isl_edges(
            CONFIG,
            positions,
            max_distance_km=MAX_ISL_DISTANCE_KM,
        )

        stats = topology_statistics(CONFIG, edge_u, edge_v, edge_d, edge_type)
        stats.update(
            {
                "time_s": float(time_s),
                "time_min": float(time_s / 60.0),
                "plane_0_to_1_best_shift": int(shifts[0]),
            }
        )
        summary_rows.append(stats)

        fixed_distance = float(
            np.linalg.norm(
                positions[0, tracked_plane_0_slot]
                - positions[1, tracked_plane_1_slot]
            )
        )
        tracked_rows.append(
            {
                "time_s": float(time_s),
                "time_min": float(time_s / 60.0),
                "plane_0_slot": tracked_plane_0_slot,
                "plane_1_slot": tracked_plane_1_slot,
                "fixed_pair_distance_km": fixed_distance,
                "within_5000_km": bool(fixed_distance <= MAX_ISL_DISTANCE_KM),
                "current_best_shift_0_to_1": int(shifts[0]),
                "fixed_pair_is_current_matching": bool(
                    tracked_plane_1_slot
                    == (tracked_plane_0_slot + shifts[0]) % CONFIG.satellites_per_plane
                ),
            }
        )

        if index % 50 == 0 or index == len(times) - 1:
            print(f"Q1 progress: {index + 1}/{len(times)}")

    summary = pd.DataFrame(summary_rows)
    tracked = pd.DataFrame(tracked_rows)

    summary.to_csv(output_dir / "topology_summary.csv", index=False, encoding="utf-8-sig")
    tracked.to_csv(output_dir / "tracked_interplane_pair.csv", index=False, encoding="utf-8-sig")

    estimated_period = estimate_dominant_period_s(
        tracked["fixed_pair_distance_km"].to_numpy(), TIME_STEP_S
    )

    with open(output_dir / "q1_conclusion.txt", "w", encoding="utf-8") as file:
        file.write(f"星座方案: M={CONFIG.planes}, N={CONFIG.satellites_per_plane}, ")
        file.write(f"i={CONFIG.inclination_deg} deg, F={CONFIG.phase_factor}\n")
        file.write(f"卫星总数: {CONFIG.satellite_count}\n")
        file.write(f"理论轨道周期: {period_s / 60.0:.6f} min\n")
        file.write(f"固定跨轨卫星对估计主周期: {estimated_period / 60.0:.6f} min\n")
        file.write(f"仿真期间网络始终连通: {bool(summary['connected'].all())}\n")
        file.write(f"仿真期间最大节点度: {int(summary['max_degree'].max())}\n")
        file.write(f"最少有效星间链路数: {int(summary['total_link_count'].min())}\n")
        file.write(f"最多有效星间链路数: {int(summary['total_link_count'].max())}\n")
        file.write(
            f"跨轨链路最大距离: {summary['max_inter_distance_km'].max():.6f} km\n"
        )
        file.write(
            "链路通断条件: 仅连接同轨前后相邻卫星和左右相邻轨道面的一一最近匹配卫星，"
            "且星间距离不超过 5000 km。\n"
        )

    plt.figure(figsize=(9, 5))
    plt.plot(tracked["time_min"], tracked["fixed_pair_distance_km"])
    plt.axhline(MAX_ISL_DISTANCE_KM, linestyle="--", label="5000 km threshold")
    plt.xlabel("Time (min)")
    plt.ylabel("Inter-plane distance (km)")
    plt.title("Periodic distance variation of a fixed inter-plane satellite pair")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fixed_interplane_distance.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(summary["time_min"], summary["intra_link_count"], label="Intra-plane")
    plt.plot(summary["time_min"], summary["inter_link_count"], label="Inter-plane")
    plt.plot(summary["time_min"], summary["total_link_count"], label="Total")
    plt.xlabel("Time (min)")
    plt.ylabel("Active link count")
    plt.title("Time-varying ISL topology")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "active_link_count.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(summary["time_min"], summary["min_inter_distance_km"], label="Minimum")
    plt.plot(summary["time_min"], summary["mean_inter_distance_km"], label="Mean")
    plt.plot(summary["time_min"], summary["max_inter_distance_km"], label="Maximum")
    plt.axhline(MAX_ISL_DISTANCE_KM, linestyle="--", label="5000 km threshold")
    plt.xlabel("Time (min)")
    plt.ylabel("Active inter-plane link distance (km)")
    plt.title("Distance range of active inter-plane links")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "interplane_link_distance_range.png", dpi=200)
    plt.close()

    print("\nQ1 finished.")
    print(f"Results saved to: {output_dir.resolve()}")
    print(f"Theoretical orbital period: {period_s / 60.0:.4f} min")
    print(f"Estimated dominant period: {estimated_period / 60.0:.4f} min")
    print(f"Always connected: {bool(summary['connected'].all())}")
    print(f"Maximum degree: {int(summary['max_degree'].max())}")


if __name__ == "__main__":
    main()
