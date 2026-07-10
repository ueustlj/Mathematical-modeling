import csv

import numpy as np

import archive.fine_validate as fv


# ============================================================
# 1. 固定参数
# ============================================================

# 轨道面数量保持不变
fv.M = 32

# 使用最终高精度
fv.grid_step = 0.5
fv.time_step = 30.0


# ============================================================
# 2. 测试范围
# ============================================================

# 从每轨65颗开始，逐步增加
N_VALUES = [
    65,
    66,
    67,
    68
]


# 采用此前精细搜索中表现最好的三组倾角和相位
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


# ============================================================
# 3. 重新生成与卫星数量有关的数组
# ============================================================

def rebuild_constellation_arrays():
    """
    修改每轨卫星数后，必须重新生成卫星编号、
    升交点赤经和面内相位数组。
    """

    fv.plane_ids = np.repeat(
        np.arange(fv.M),
        fv.N
    )

    fv.satellite_ids = np.tile(
        np.arange(fv.N),
        fv.M
    )

    fv.raan = (
        2.0
        * np.pi
        * fv.plane_ids
        / fv.M
    )

    fv.phase_in_plane = (
        2.0
        * np.pi
        * fv.satellite_ids
        / fv.N
    )


# ============================================================
# 4. 结果排序
# ============================================================

def ranking_key(result):
    """
    优先级：
    1. 可行方案优先；
    2. 总卫星数越少越好；
    3. 全区域覆盖时间比例越高越好；
    4. 最差时刻覆盖率越高越好；
    5. 最大覆盖间隙越小越好。
    """

    return (
        0 if result["feasible"] else 1,

        result["total_satellites"],

        -result[
            "full_region_coverage_ratio"
        ],

        -result[
            "worst_instantaneous_ratio"
        ],

        result[
            "maximum_gap_minutes"
        ]
    )


# ============================================================
# 5. 主程序
# ============================================================

def main():

    all_results = []

    first_feasible_n = None

    print("=" * 75)
    print("开始测试增加面内卫星数量后的高精度覆盖性能")
    print("=" * 75)

    for satellites_per_plane in N_VALUES:

        fv.N = satellites_per_plane

        rebuild_constellation_arrays()

        print("\n")
        print("#" * 75)

        print(
            f"开始测试：M={fv.M}，"
            f"N={fv.N}，"
            f"卫星总数={fv.M * fv.N}"
        )

        print("#" * 75)

        current_n_feasible = False

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
            print("-" * 75)

            print(
                f"候选 {candidate_id}："
                f"i={fv.inclination:.2f}°，"
                f"F={fv.walker_phase_factor}"
            )

            print("-" * 75)

            (
                results,
                point_coverage_grid,
                latitude_grid,
                longitude_grid,
                elapsed_time
            ) = fv.run_fine_validation()

            one_result = {
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
            }

            all_results.append(
                one_result
            )

            print("\n本组结果：")

            print(
                "时空覆盖率："
                f"{100 * one_result['space_time_coverage_ratio']:.6f}%"
            )

            print(
                "全区域覆盖时间比例："
                f"{100 * one_result['full_region_coverage_ratio']:.6f}%"
            )

            print(
                "最差时刻覆盖率："
                f"{100 * one_result['worst_instantaneous_ratio']:.6f}%"
            )

            print(
                "最大覆盖间隙："
                f"{one_result['maximum_gap_minutes']:.2f} min"
            )

            print(
                "是否通过："
                f"{one_result['feasible']}"
            )

            if one_result["feasible"]:

                current_n_feasible = True

        # 当前N已经存在可行方案，
        # 不再继续测试更大的卫星数量
        if current_n_feasible:

            first_feasible_n = fv.N

            print("\n")
            print("*" * 75)

            print(
                f"发现高精度可行规模："
                f"M={fv.M}，N={fv.N}，"
                f"总卫星数={fv.M * fv.N}"
            )

            print("*" * 75)

            break

    # ========================================================
    # 保存与输出结果
    # ========================================================

    sorted_results = sorted(
        all_results,
        key=ranking_key
    )

    filename = (
        "expanded_satellite_test.csv"
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
                sorted_results[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            sorted_results
        )

    print("\n")
    print("=" * 135)

    print(
        "排名  M   N   总星数   倾角    F   "
        "时空覆盖率   全区域覆盖时间比例   "
        "最差时刻覆盖率   最差地点覆盖率   "
        "平均覆盖重数   最大间隙/min   可行"
    )

    print("=" * 135)

    for rank, result in enumerate(
            sorted_results,
            start=1
    ):

        print(
            f"{rank:>4} "
            f"{result['M']:>3} "
            f"{result['N']:>3} "
            f"{result['total_satellites']:>8} "
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

    print("=" * 135)

    print(
        f"\n结果已保存到：{filename}"
    )

    feasible_results = [
        result
        for result in sorted_results
        if result["feasible"]
    ]

    if feasible_results:

        best = feasible_results[0]

        print("\n当前卫星数量搜索得到的最优可行方案：")

        print(
            f"M = {best['M']}"
        )

        print(
            f"N = {best['N']}"
        )

        print(
            f"i = {best['inclination']:.2f}°"
        )

        print(
            f"F = {best['phase_factor']}"
        )

        print(
            f"卫星总数 = "
            f"{best['total_satellites']}"
        )

        print(
            "全区域连续覆盖时间比例 = "
            f"{100 * best['full_region_coverage_ratio']:.6f}%"
        )

        print(
            "最大覆盖间隙 = "
            f"{best['maximum_gap_minutes']:.2f} min"
        )

    else:

        print(
            "\nN=65～68的这些候选参数仍未通过。"
        )

        print(
            "下一步需要扩大相位因子和倾角搜索范围。"
        )


if __name__ == "__main__":
    main()