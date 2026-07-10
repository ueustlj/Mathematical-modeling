from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from problem2_model import (
    ConstellationConfig,
    ConstellationEvaluator,
    SimulationResolution,
)


# ============================================================
# 只需要修改这里的候选星座参数
# ============================================================
CANDIDATE = ConstellationConfig(
    planes=38,
    satellites_per_plane=44,
    inclination_deg=53.0,
    phase_factor=1,
    raan_offset_deg=0.0,
    initial_phase_deg=0.0,
)

# 中等精度验证：约数秒到数十秒，先用它判断方案是否接近可行。
RESOLUTION = SimulationResolution(
    grid_step_deg=0.5,
    time_step_s=30.0,
    duration_s=24.0 * 3600.0,
)

# 最终论文验证建议改为：
# grid_step_deg=0.5, time_step_s=30.0, duration_s=72.0*3600.0
# 这会明显增加运行时间。

OUTPUT_DIR = Path("results")


def save_summary(result) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "validation_summary.csv"

    row = result.to_dict()
    with output_file.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print(f"结果已保存：{output_file.resolve()}")


def draw_heatmaps(grids) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    extent = [
        grids["longitudes"][0],
        grids["longitudes"][-1],
        grids["latitudes"][0],
        grids["latitudes"][-1],
    ]

    # 单重覆盖缺失率比 0~100% 覆盖率图更容易暴露小漏洞。
    missing_percent = 100.0 * (1.0 - grids["single_time_ratio_grid"])
    plt.figure(figsize=(10, 7))
    image = plt.imshow(
        missing_percent,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    plt.colorbar(image, label="Single-coverage missing time (%)")
    plt.xlabel("Longitude (degree)")
    plt.ylabel("Latitude (degree)")
    plt.title("Single-Coverage Missing-Time Map")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "single_missing_time_map.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 7))
    image = plt.imshow(
        100.0 * grids["double_time_ratio_grid"],
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=0.0,
        vmax=100.0,
    )
    plt.colorbar(image, label="Double-coverage time ratio (%)")
    plt.xlabel("Longitude (degree)")
    plt.ylabel("Latitude (degree)")
    plt.title("Double-Coverage Time Ratio")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "double_coverage_time_map.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 7))
    image = plt.imshow(
        grids["maximum_single_gap_grid_minutes"],
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    plt.colorbar(image, label="Maximum uncovered gap (min)")
    plt.xlabel("Longitude (degree)")
    plt.ylabel("Latitude (degree)")
    plt.title("Maximum Single-Coverage Gap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "maximum_gap_map.png", dpi=300)
    plt.close()


def print_result(result) -> None:
    print("\n" + "=" * 68)
    print("问题二候选星座验证结果")
    print("=" * 68)
    print(f"M = {result.planes}")
    print(f"N = {result.satellites_per_plane}")
    print(f"i = {result.inclination_deg:.2f}°")
    print(f"F = {result.phase_factor}")
    print(f"卫星总数 = {result.total_satellites}")
    print(f"单重时空覆盖率 = {100*result.single_space_time_ratio:.6f}%")
    print(
        "全区域单重覆盖时间比例 = "
        f"{100*result.single_full_region_time_ratio:.6f}%"
    )
    print(f"最大单重覆盖间隙 = {result.maximum_single_gap_minutes:.2f} min")
    print(f"最差时刻区域覆盖率 = {100*result.worst_single_instantaneous_ratio:.6f}%")
    print(f"平均覆盖重数 = {result.average_multiplicity:.4f}")
    print(
        "全区域二重覆盖时间比例 = "
        f"{100*result.double_full_region_time_ratio:.6f}%"
    )
    print(f"问题二第(2)问是否通过 = {result.single_feasible}")
    print(f"问题二第(3)问是否通过 = {result.double_95_feasible}")
    print(f"发射次数 = {result.launch_count}")
    print(f"估算总成本 = {result.total_cost_yuan/1e8:.2f} 亿元")
    print("=" * 68)


def main() -> None:
    evaluator = ConstellationEvaluator(RESOLUTION)
    result, grids = evaluator.evaluate(
        CANDIDATE,
        return_grids=True,
        progress=True,
    )
    print_result(result)
    save_summary(result)
    if grids is not None:
        draw_heatmaps(grids)
        print(f"图像已保存到：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
