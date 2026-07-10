import csv
import time

import numpy as np

from q2_3_double_coverage import (
    calculate_satellite_vectors,
    COS_PSI_MAX
)


# ============================================================
# 1. 原单重覆盖最优方案
# ============================================================

BASE_M = 36
BASE_N = 44
BASE_I = 53.0
BASE_F = 1

BASE_TOTAL = BASE_M * BASE_N


# ============================================================
# 2. 建立用于快速搜索的较粗地面网格
# ============================================================

def create_fast_ground_grid(
    latitude_number=15,
    longitude_number=19
):
    """
    搜索阶段使用较粗网格，以提高计算速度。

    纬度范围：4°N ~ 53°N
    经度范围：73°E ~ 135°E
    """

    latitude_deg = np.linspace(
        4,
        53,
        latitude_number
    )

    longitude_deg = np.linspace(
        73,
        135,
        longitude_number
    )

    longitude_grid, latitude_grid = np.meshgrid(
        longitude_deg,
        latitude_deg
    )

    latitude = np.deg2rad(
        latitude_grid.ravel()
    )

    longitude = np.deg2rad(
        longitude_grid.ravel()
    )

    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)

    ground_vectors = np.column_stack(
        (x, y, z)
    )

    return ground_vectors


# ============================================================
# 3. 快速评价函数
# ============================================================

def fast_evaluate_constellation(
    M,
    N,
    inclination_deg,
    F,
    simulation_hours=24,
    time_step_minutes=60
):
    """
    搜索阶段快速评价。

    使用：
    1. 较粗地面网格；
    2. 60 分钟时间步长。

    搜索得到候选方案后，
    还要再使用精细模型进行验证。
    """

    ground_vectors = create_fast_ground_grid()

    time_step_seconds = time_step_minutes * 60

    # 不包含第 24 小时，避免和第 0 小时近似重复
    times = np.arange(
        0,
        simulation_hours * 3600,
        time_step_seconds
    )

    single_success_number = 0
    strict_double_success_number = 0

    space_time_double_sum = 0.0
    average_cover_sum = 0.0

    global_minimum_cover = 10 ** 9

    for time_s in times:

        satellite_vectors = calculate_satellite_vectors(
            M=M,
            N=N,
            inclination_deg=inclination_deg,
            F=F,
            time_s=time_s
        )

        cosine_matrix = (
            ground_vectors
            @ satellite_vectors.T
        )

        covered_matrix = (
            cosine_matrix >= COS_PSI_MAX
        )

        cover_count = np.sum(
            covered_matrix,
            axis=1
        )

        current_minimum = int(
            np.min(cover_count)
        )

        global_minimum_cover = min(
            global_minimum_cover,
            current_minimum
        )

        if current_minimum >= 1:
            single_success_number += 1

        if current_minimum >= 2:
            strict_double_success_number += 1

        space_time_double_sum += np.mean(
            cover_count >= 2
        )

        average_cover_sum += np.mean(
            cover_count
        )

    time_number = len(times)

    single_ratio = (
        single_success_number
        / time_number
    )

    strict_double_ratio = (
        strict_double_success_number
        / time_number
    )

    space_time_double_ratio = (
        space_time_double_sum
        / time_number
    )

    average_multiplicity = (
        average_cover_sum
        / time_number
    )

    return {
        "single_ratio": single_ratio,
        "strict_double_ratio": strict_double_ratio,
        "space_time_double_ratio": space_time_double_ratio,
        "average_multiplicity": average_multiplicity,
        "minimum_cover": global_minimum_cover
    }


# ============================================================
# 4. 搜索 M 和 N
# ============================================================

def search_M_N():

    print("=" * 75)
    print("问题二第（3）问：第一阶段搜索 M 和 N")
    print("=" * 75)

    print(
        f"原单重覆盖方案："
        f"M={BASE_M}, N={BASE_N}, "
        f"i={BASE_I:.1f}°, F={BASE_F}"
    )

    print(f"原卫星总数：{BASE_TOTAL}")
    print()

    # 在原最优方案附近搜索
    M_values = range(34, 43)   # 34 ~ 42
    N_values = range(42, 53)   # 42 ~ 52

    candidate_list = []

    for M in M_values:
        for N in N_values:

            total_satellites = M * N

            # 二重覆盖方案不可能比单重最优方案卫星更少
            if total_satellites < BASE_TOTAL:
                continue

            candidate_list.append(
                (total_satellites, M, N)
            )

    # 按卫星总数从少到多搜索
    candidate_list.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2]
        )
    )

    all_results = []
    feasible_results = []

    best_total = None

    start_time = time.time()

    for index, candidate in enumerate(
        candidate_list,
        start=1
    ):

        total_satellites, M, N = candidate

        # 已找到更小总星数的可行解后，
        # 不再搜索卫星更多的方案
        if (
            best_total is not None
            and total_satellites > best_total
        ):
            break

        print(
            f"[{index:03d}/{len(candidate_list):03d}] "
            f"测试 M={M:2d}, N={N:2d}, "
            f"总星数={total_satellites:4d} ... ",
            end="",
            flush=True
        )

        result = fast_evaluate_constellation(
            M=M,
            N=N,
            inclination_deg=BASE_I,
            F=BASE_F,
            simulation_hours=24,
            time_step_minutes=60
        )

        feasible = (
            result["single_ratio"] >= 0.999999
            and
            result["strict_double_ratio"] >= 0.95
        )

        print(
            f"单重={result['single_ratio'] * 100:6.2f}%  "
            f"严格二重="
            f"{result['strict_double_ratio'] * 100:6.2f}%  "
            f"时空二重="
            f"{result['space_time_double_ratio'] * 100:6.2f}%  "
            f"最低={result['minimum_cover']}",
            end=""
        )

        if feasible:
            print("  ← 可行")
        else:
            print()

        row = {
            "M": M,
            "N": N,
            "inclination_deg": BASE_I,
            "F": BASE_F,
            "total_satellites": total_satellites,
            "single_ratio": result["single_ratio"],
            "strict_double_ratio":
                result["strict_double_ratio"],
            "space_time_double_ratio":
                result["space_time_double_ratio"],
            "average_multiplicity":
                result["average_multiplicity"],
            "minimum_cover":
                result["minimum_cover"],
            "feasible": feasible
        }

        all_results.append(row)

        if feasible:

            feasible_results.append(row)

            if best_total is None:
                best_total = total_satellites

    elapsed_time = time.time() - start_time

    # 保存搜索结果
    output_filename = "q2_3_stage1_search_results.csv"

    with open(
        output_filename,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        fieldnames = [
            "M",
            "N",
            "inclination_deg",
            "F",
            "total_satellites",
            "single_ratio",
            "strict_double_ratio",
            "space_time_double_ratio",
            "average_multiplicity",
            "minimum_cover",
            "feasible"
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print()
    print("=" * 75)
    print("第一阶段搜索完成")
    print("=" * 75)

    print(
        f"搜索耗时：{elapsed_time:.2f} 秒"
    )

    print(
        f"结果已保存至：{output_filename}"
    )

    if len(feasible_results) == 0:

        print()
        print("当前搜索范围内没有找到满足条件的方案。")
        print("下一步需要扩大 M、N 的搜索范围。")

        return None

    feasible_results.sort(
        key=lambda row: (
            row["total_satellites"],
            -row["strict_double_ratio"],
            -row["space_time_double_ratio"]
        )
    )

    print()
    print("最小卫星总数下的可行方案：")

    for row in feasible_results:

        print(
            f"M={row['M']}, "
            f"N={row['N']}, "
            f"总星数={row['total_satellites']}, "
            f"严格二重="
            f"{row['strict_double_ratio'] * 100:.2f}%, "
            f"时空二重="
            f"{row['space_time_double_ratio'] * 100:.2f}%, "
            f"平均重数="
            f"{row['average_multiplicity']:.3f}"
        )

    best_result = feasible_results[0]

    print()
    print("-" * 75)
    print("第一阶段推荐候选方案")
    print("-" * 75)

    print(f"M = {best_result['M']}")
    print(f"N = {best_result['N']}")
    print(f"i = {best_result['inclination_deg']:.1f}°")
    print(f"F = {best_result['F']}")

    print(
        f"卫星总数 = "
        f"{best_result['total_satellites']}"
    )

    print(
        f"相比原方案增加 = "
        f"{best_result['total_satellites'] - BASE_TOTAL} 颗"
    )

    print(
        f"卫星数量增加比例 = "
        f"{(
            best_result['total_satellites']
            / BASE_TOTAL
            - 1
        ) * 100:.2f}%"
    )

    return best_result


# ============================================================
# 5. 主程序
# ============================================================

if __name__ == "__main__":

    search_M_N()