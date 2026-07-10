import matplotlib.pyplot as plt

from archive.parameter import (
    M,
    N,
    inclination,
    latitude_min,
    latitude_max,
    longitude_min,
    longitude_max
)

from archive.coverage import evaluate_constellation


def print_results(results):
    """
    输出星座覆盖评估结果。
    """

    print("\n")
    print("=" * 50)
    print("低轨卫星星座覆盖性能评估结果")
    print("=" * 50)

    print(
        f"轨道面数量："
        f"{results['plane_number']}"
    )

    print(
        f"每轨卫星数量："
        f"{results['satellites_per_plane']}"
    )

    print(
        f"卫星总数："
        f"{results['total_satellites']}"
    )

    print(
        f"轨道倾角："
        f"{results['inclination']:.1f}°"
    )

    print(
        "时空覆盖率："
        f"{100.0 * results['space_time_coverage_ratio']:.4f}%"
    )

    print(
        "全区域连续覆盖时间比例："
        f"{100.0 * results['full_region_coverage_ratio']:.4f}%"
    )

    print(
        "平均覆盖重数："
        f"{results['average_multiplicity']:.4f}"
    )

    print(
        "最大覆盖间隙："
        f"{results['maximum_gap_minutes']:.2f} min"
    )

    print("=" * 50)


def draw_coverage_heatmap(results):
    """
    绘制目标区域内各地面点的时间覆盖率热力图。
    """

    coverage_grid = (
        100.0
        * results[
            "point_time_coverage_grid"
        ]
    )

    plt.figure(
        figsize=(10, 7)
    )

    image = plt.imshow(
        coverage_grid,
        origin="lower",
        extent=[
            longitude_min,
            longitude_max,
            latitude_min,
            latitude_max
        ],
        aspect="auto",
        vmin=0,
        vmax=100
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
        "Regional Time Coverage Ratio"
    )

    plt.tight_layout()

    plt.savefig(
        "coverage_heatmap.png",
        dpi=300
    )

    plt.show()


def main():
    """
    主程序入口。
    """

    print("开始进行24小时区域覆盖仿真……")

    results = evaluate_constellation(
        plane_number=M,
        satellites_per_plane=N,
        inclination_degree=inclination
    )

    print_results(results)

    draw_coverage_heatmap(results)


if __name__ == "__main__":
    main()