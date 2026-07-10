from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problem2_model import (
    ConstellationConfig,
    ConstellationEvaluator,
    SimulationResolution,
)


# ============================================================
# 把 search_problem2_double.py 输出的最优候选填在这里
# ============================================================
CANDIDATE = ConstellationConfig(
    planes=41,
    satellites_per_plane=48,
    inclination_deg=53.0,
    phase_factor=26,
    raan_offset_deg=0.0,
    initial_phase_deg=0.0,
)

# 先用这一档确认程序和候选是否正常。
RESOLUTION = SimulationResolution(
    grid_step_deg=0.5,
    time_step_s=30.0,
    duration_s=24.0 * 3600.0,
)

# 最终论文验证可改为：
# RESOLUTION = SimulationResolution(
#     grid_step_deg=0.5,
#     time_step_s=30.0,
#     duration_s=72.0 * 3600.0,
# )
#
# 0.25°、10 s、72 h 会非常慢，不建议一开始直接运行。

OUTPUT_DIR = Path("results_double_validation")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_summary(result) -> None:
    row = result.to_dict()
    row["joint_feasible"] = bool(
        result.single_feasible and result.double_95_feasible
    )

    output_file = OUTPUT_DIR / "double_validation_summary.csv"
    with output_file.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print(f"汇总结果已保存：{output_file.resolve()}")


def draw_double_heatmap(grids) -> None:
    extent = [
        grids["longitudes"][0],
        grids["longitudes"][-1],
        grids["latitudes"][0],
        grids["latitudes"][-1],
    ]

    double_ratio_percent = 100.0 * grids["double_time_ratio_grid"]
    double_missing_percent = 100.0 - double_ratio_percent

    plt.figure(figsize=(10, 7))
    image = plt.imshow(
        double_ratio_percent,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=0.0,
        vmax=100.0,
    )
    plt.colorbar(image, label="Double-coverage time ratio (%)")
    plt.xlabel("Longitude (degree)")
    plt.ylabel("Latitude (degree)")
    plt.title("Pointwise Double-Coverage Time Ratio")
    plt.tight_layout()
    path = OUTPUT_DIR / "double_coverage_time_ratio_map.png"
    plt.savefig(path, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 7))
    image = plt.imshow(
        double_missing_percent,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    plt.colorbar(image, label="Time without double coverage (%)")
    plt.xlabel("Longitude (degree)")
    plt.ylabel("Latitude (degree)")
    plt.title("Pointwise Double-Coverage Deficiency")
    plt.tight_layout()
    path2 = OUTPUT_DIR / "double_coverage_deficiency_map.png"
    plt.savefig(path2, dpi=300)
    plt.close()

    weakest_index = int(np.argmin(double_ratio_percent))
    row, col = np.unravel_index(
        weakest_index,
        double_ratio_percent.shape,
    )
    weakest_lat = grids["latitudes"][row]
    weakest_lon = grids["longitudes"][col]
    weakest_ratio = double_ratio_percent[row, col]

    print(
        "点态二重覆盖比例最低的位置："
        f"lat={weakest_lat:.3f}°, "
        f"lon={weakest_lon:.3f}°, "
        f"ratio={weakest_ratio:.6f}%"
    )
    print(f"热力图已保存：{path.resolve()}")
    print(f"缺失图已保存：{path2.resolve()}")


def print_result(result) -> None:
    joint = bool(result.single_feasible and result.double_95_feasible)

    print("\n" + "=" * 76)
    print("第二题第（3）小问：严格二重覆盖验证")
    print("=" * 76)
    print(f"M = {result.planes}")
    print(f"N = {result.satellites_per_plane}")
    print(f"i = {result.inclination_deg:.2f}°")
    print(f"F = {result.phase_factor}")
    print(f"卫星总数 = {result.total_satellites}")
    print(f"网格步长 = {result.grid_step_deg:.3f}°")
    print(f"时间步长 = {result.time_step_s:.1f} s")
    print(f"仿真时长 = {result.duration_h:.1f} h")
    print("-" * 76)
    print(
        "全区域二重覆盖时间比例 = "
        f"{100.0*result.double_full_region_time_ratio:.8f}%"
    )
    print(
        "二重覆盖时空比例（辅助指标） = "
        f"{100.0*result.double_space_time_ratio:.8f}%"
    )
    print(f"平均覆盖重数 = {result.average_multiplicity:.6f}")
    print(
        "全区域单重覆盖时间比例 = "
        f"{100.0*result.single_full_region_time_ratio:.8f}%"
    )
    print("-" * 76)
    print(f"连续单重覆盖通过 = {result.single_feasible}")
    print(f"95% 时间全区域二重覆盖通过 = {result.double_95_feasible}")
    print(f"联合约束通过 = {joint}")
    print(f"发射次数 = {result.launch_count}")
    print(f"制造成本 = {result.manufacturing_cost_yuan / 1e8:.4f} 亿元")
    print(f"发射成本 = {result.launch_cost_yuan / 1e8:.4f} 亿元")
    print(f"总成本 = {result.total_cost_yuan / 1e8:.4f} 亿元")

    if not result.double_95_feasible:
        shortfall = max(
            0.0,
            95.0 - 100.0 * result.double_full_region_time_ratio,
        )
        print(f"距离 95% 还差 = {shortfall:.8f} 个百分点")


def main() -> None:
    evaluator = ConstellationEvaluator(RESOLUTION)
    result, grids = evaluator.evaluate(
        CANDIDATE,
        return_grids=True,
        progress=True,
    )

    print_result(result)
    save_summary(result)

    if grids is None:
        raise RuntimeError("return_grids=True，但没有返回网格结果")

    draw_double_heatmap(grids)


if __name__ == "__main__":
    main()
