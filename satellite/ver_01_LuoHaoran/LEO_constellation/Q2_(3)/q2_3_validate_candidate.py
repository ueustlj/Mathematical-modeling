from q2_3_double_coverage import evaluate_constellation


# ============================================================
# 第一阶段搜索得到的候选方案
# ============================================================

M = 41
N = 47
INCLINATION = 53.0
F = 1


if __name__ == "__main__":

    print("=" * 65)
    print("问题二第（3）问：候选方案精细验证")
    print("=" * 65)

    print(f"轨道面数 M：{M}")
    print(f"每轨卫星数 N：{N}")
    print(f"卫星总数：{M * N}")
    print(f"轨道倾角：{INCLINATION:.1f}°")
    print(f"相位参数 F：{F}")
    print()

    print("正在进行精细覆盖仿真……")
    print("空间网格：26 × 32")
    print("仿真时间：24 小时")
    print("时间步长：10 分钟")
    print()

    result = evaluate_constellation(
        M=M,
        N=N,
        inclination_deg=INCLINATION,
        F=F,
        simulation_hours=24,
        time_step_minutes=10
    )

    print()
    print("=" * 65)
    print("精细仿真结果")
    print("=" * 65)

    print(
        "单重连续覆盖时间比例："
        f"{result['single_coverage_time_ratio'] * 100:.2f}%"
    )

    print(
        "严格二重覆盖时间比例："
        f"{result['strict_double_coverage_time_ratio'] * 100:.2f}%"
    )

    print(
        "时空网格二重覆盖比例："
        f"{result['space_time_double_ratio'] * 100:.2f}%"
    )

    print(
        "平均覆盖重数："
        f"{result['average_multiplicity']:.3f}"
    )

    print(
        "全程最低覆盖重数："
        f"{result['global_minimum_cover']}"
    )

    print(
        "最薄弱时刻："
        f"{result['worst_time_hours']:.2f} 小时"
    )

    print("=" * 65)

    single_ok = (
        result["single_coverage_time_ratio"]
        >= 0.999999
    )

    double_ok = (
        result["strict_double_coverage_time_ratio"]
        >= 0.95
    )

    if single_ok and double_ok:

        print("精细验证结论：候选方案通过。")

    elif single_ok:

        print(
            "精细验证结论：保持单重连续覆盖，"
            "但严格二重覆盖率未达到 95%。"
        )

    else:

        print(
            "精细验证结论：该方案在精细网格下"
            "未能保持单重连续覆盖。"
        )