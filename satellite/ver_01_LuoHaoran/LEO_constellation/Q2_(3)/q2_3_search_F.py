import csv
import time

from q2_3_pointwise_validate import (
    evaluate_pointwise_double_coverage
)


# ============================================================
# 1. 当前候选方案
# ============================================================

# 若刚才测试的是 41 × 48，保持下面参数
M = 41
N = 48

# 若刚才测试的是 42 × 47，只修改成：
# M = 42
# N = 47

INCLINATION = 53.0


# ============================================================
# 2. 搜索全部相位参数 F
# ============================================================

def search_all_F():

    print("=" * 85)
    print("问题二第（3）问：相位参数 F 精细搜索")
    print("=" * 85)

    print(f"M = {M}")
    print(f"N = {N}")
    print(f"总卫星数 = {M * N}")
    print(f"倾角 = {INCLINATION:.2f}°")
    print(f"搜索范围：F = 0 ~ {M - 1}")
    print()

    results = []

    start_time = time.time()

    for F in range(M):

        print(
            f"[{F + 1:02d}/{M:02d}] "
            f"正在测试 F={F:2d} ... ",
            end="",
            flush=True
        )

        result = evaluate_pointwise_double_coverage(
            M=M,
            N=N,
            inclination_deg=INCLINATION,
            F=F,
            simulation_hours=24,
            time_step_minutes=10
        )

        worst_single_ratio = result["worst_single_ratio"]
        worst_double_ratio = result["worst_double_ratio"]
        average_double_ratio = result["average_double_ratio"]
        qualified_point_ratio = result["qualified_point_ratio"]
        strict_double_ratio = result["strict_double_ratio"]
        average_multiplicity = result["average_multiplicity"]

        feasible = (
            worst_single_ratio >= 0.999999
            and worst_double_ratio >= 0.95
        )

        print(
            f"最弱点二重={worst_double_ratio * 100:6.2f}%  "
            f"达标网格={qualified_point_ratio * 100:6.2f}%  "
            f"全区同时二重={strict_double_ratio * 100:6.2f}%"
            + ("  ← 通过" if feasible else "")
        )

        results.append({
            "M": M,
            "N": N,
            "total_satellites": M * N,
            "inclination_deg": INCLINATION,
            "F": F,
            "worst_single_ratio": worst_single_ratio,
            "worst_double_ratio": worst_double_ratio,
            "average_double_ratio": average_double_ratio,
            "qualified_point_ratio": qualified_point_ratio,
            "strict_double_ratio": strict_double_ratio,
            "average_multiplicity": average_multiplicity,
            "worst_latitude": result["worst_latitude"],
            "worst_longitude": result["worst_longitude"],
            "feasible": feasible
        })

    elapsed_time = time.time() - start_time

    # ========================================================
    # 3. 保存结果
    # ========================================================

    output_filename = "q2_3_F_search_results.csv"

    with open(
        output_filename,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        fieldnames = list(results[0].keys())

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    # 优先比较最弱地点二重覆盖率
    results.sort(
        key=lambda row: (
            -row["worst_double_ratio"],
            -row["qualified_point_ratio"],
            -row["strict_double_ratio"],
            -row["average_double_ratio"]
        )
    )

    feasible_results = [
        row for row in results
        if row["feasible"]
    ]

    print()
    print("=" * 85)
    print("搜索完成")
    print("=" * 85)

    print(f"耗时：{elapsed_time:.2f} 秒")
    print(f"结果文件：{output_filename}")

    print()
    print("表现最好的 10 个相位参数：")
    print("-" * 85)

    for rank, row in enumerate(
        results[:10],
        start=1
    ):
        print(
            f"{rank:2d}. "
            f"F={row['F']:2d}，"
            f"最弱点二重="
            f"{row['worst_double_ratio'] * 100:6.2f}%，"
            f"达标网格="
            f"{row['qualified_point_ratio'] * 100:6.2f}%，"
            f"全区同时二重="
            f"{row['strict_double_ratio'] * 100:6.2f}%，"
            f"最弱点位置="
            f"({row['worst_latitude']:.2f}°N, "
            f"{row['worst_longitude']:.2f}°E)"
        )

    print()
    print("=" * 85)

    if feasible_results:

        best = feasible_results[0]

        print("找到满足要求的方案：")
        print(f"M = {best['M']}")
        print(f"N = {best['N']}")
        print(f"i = {best['inclination_deg']:.2f}°")
        print(f"F = {best['F']}")

        print(
            "最弱地点二重覆盖时间比例 = "
            f"{best['worst_double_ratio'] * 100:.2f}%"
        )

        print(
            "达到 95% 二重覆盖的网格点比例 = "
            f"{best['qualified_point_ratio'] * 100:.2f}%"
        )

        print(
            "全区域同时二重覆盖时间比例 = "
            f"{best['strict_double_ratio'] * 100:.2f}%"
        )

        print("结论：当前卫星数量下，通过调整相位即可满足要求。")

    else:

        best = results[0]

        print("当前倾角下尚未找到完全达标的相位。")
        print()
        print("当前最优结果：")
        print(f"F = {best['F']}")

        print(
            "最弱地点二重覆盖时间比例 = "
            f"{best['worst_double_ratio'] * 100:.2f}%"
        )

        print(
            "最弱地点位置 = "
            f"{best['worst_latitude']:.2f}°N，"
            f"{best['worst_longitude']:.2f}°E"
        )

        print("下一步应围绕该 F 微调轨道倾角。")


if __name__ == "__main__":
    search_all_F()