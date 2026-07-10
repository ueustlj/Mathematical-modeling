import csv

import archive.fine_validate as fv


# ============================================================
# 高精度仿真设置
# ============================================================

fv.M = 32
fv.N = 64

fv.grid_step = 0.5
fv.time_step = 30.0


# 在 1° / 60 s 搜索中通过的三组方案
CANDIDATES = [
    {
        "inclination": 53.00,
        "phase_factor": 23
    },
    {
        "inclination": 53.50,
        "phase_factor": 21
    },
    {
        "inclination": 53.75,
        "phase_factor": 21
    }
]


def main():

    comparison_results = []

    print("=" * 70)
    print("开始进行三组候选星座的高精度对比验证")
    print("=" * 70)

    for candidate_id, candidate in enumerate(
            CANDIDATES,
            start=1
    ):

        fv.inclination = candidate[
            "inclination"
        ]

        fv.walker_phase_factor = candidate[
            "phase_factor"
        ]

        print("\n")
        print("#" * 70)

        print(
            f"正在验证第 {candidate_id} 组："
        )

        print(
            f"M = {fv.M}, "
            f"N = {fv.N}, "
            f"i = {fv.inclination:.2f}°, "
            f"F = {fv.walker_phase_factor}"
        )

        print("#" * 70)

        (
            results,
            point_coverage_grid,
            latitude_grid,
            longitude_grid,
            elapsed_time
        ) = fv.run_fine_validation()

        comparison_results.append({
            "candidate_id":
                candidate_id,

            "M":
                fv.M,

            "N":
                fv.N,

            "total_satellites":
                fv.M * fv.N,

            "inclination":
                fv.inclination,

            "phase_factor":
                fv.walker_phase_factor,

            "space_time_coverage_ratio":
                results[
                    "space_time_coverage_ratio"
                ],

            "full_region_coverage_ratio":
                results[
                    "full_region_coverage_ratio"
                ],

            "worst_instantaneous_ratio":
                results[
                    "worst_instantaneous_ratio"
                ],

            "minimum_point_ratio":
                results[
                    "minimum_point_ratio"
                ],

            "average_multiplicity":
                results[
                    "average_multiplicity"
                ],

            "maximum_gap_minutes":
                results[
                    "maximum_gap_minutes"
                ],

            "feasible":
                results[
                    "feasible"
                ],

            "elapsed_time_second":
                elapsed_time
        })

    # 排序：可行方案优先，其次按覆盖性能排序
    comparison_results.sort(
        key=lambda result: (
            0 if result["feasible"] else 1,

            -result[
                "full_region_coverage_ratio"
            ],

            -result[
                "worst_instantaneous_ratio"
            ],

            -result[
                "space_time_coverage_ratio"
            ],

            result[
                "maximum_gap_minutes"
            ]
        )
    )

    print("\n")
    print("=" * 120)

    print(
        "排名  倾角    F   时空覆盖率   "
        "全区域覆盖时间比例   最差时刻覆盖率   "
        "最差地点覆盖率   平均覆盖重数   "
        "最大间隙/min   可行"
    )

    print("=" * 120)

    for rank, result in enumerate(
            comparison_results,
            start=1
    ):

        print(
            f"{rank:>4} "
            f"{result['inclination']:>6.2f} "
            f"{result['phase_factor']:>3} "
            f"{100 * result['space_time_coverage_ratio']:>11.6f}% "
            f"{100 * result['full_region_coverage_ratio']:>18.6f}% "
            f"{100 * result['worst_instantaneous_ratio']:>16.6f}% "
            f"{100 * result['minimum_point_ratio']:>16.6f}% "
            f"{result['average_multiplicity']:>14.6f} "
            f"{result['maximum_gap_minutes']:>13.2f} "
            f"{str(result['feasible']):>7}"
        )

    print("=" * 120)

    filename = (
        "high_precision_candidate_comparison.csv"
    )

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                comparison_results[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            comparison_results
        )

    best = comparison_results[0]

    print("\n当前高精度最优候选：")

    print(
        f"i = {best['inclination']:.2f}°"
    )

    print(
        f"F = {best['phase_factor']}"
    )

    print(
        "全区域连续覆盖时间比例 = "
        f"{100 * best['full_region_coverage_ratio']:.6f}%"
    )

    print(
        "最大覆盖间隙 = "
        f"{best['maximum_gap_minutes']:.2f} min"
    )

    print(
        f"高精度验证是否通过："
        f"{best['feasible']}"
    )

    print(
        f"\n结果已保存至：{filename}"
    )


if __name__ == "__main__":
    main()