"""
问题四第（1）问：
单颗卫星碰撞概率与年度避撞决策模型

输出文件：
1. Q4_1_summary.txt
2. Q4_1_threshold_analysis.csv
3. Q4_1_threshold_curve.png

所需库：
numpy
pandas
matplotlib
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. 可修改的模型参数
# ============================================================

# 随机种子，保证每次运行结果可以复现
RANDOM_SEED = 20260710

# Monte Carlo样本数
# 若电脑运行较慢，可改为200000
N_MONTE_CARLO = 500_000

# ------------------------------------------------------------
# 空间碎片环境参数
# ------------------------------------------------------------

# 直径大于1 cm碎片的数密度，单位：个/km^3
DEBRIS_DENSITY = 1.0e-8

# 碎片直径范围，单位：m
DEBRIS_DIAMETER_MIN = 0.01
DEBRIS_DIAMETER_MAX = 1.00

# 碎片尺寸幂律分布指数
# f(d) = C * d^(-beta)
SIZE_POWER_BETA = 3.0

# ------------------------------------------------------------
# 卫星外形参数
# ------------------------------------------------------------

# 单颗卫星等效迎风面积，单位：m^2
# 题目未给定尺寸，因此将其作为合理假设
SATELLITE_AREA_M2 = 20.0

# ------------------------------------------------------------
# 相对速度参数
# ------------------------------------------------------------

# 相对速度均值，单位：km/s
RELATIVE_SPEED_MEAN = 10.0

# 相对速度标准差，单位：km/s
RELATIVE_SPEED_STD = 2.0

# ------------------------------------------------------------
# 危险交会筛选参数
# ------------------------------------------------------------

# 预警筛选半径，单位：km
WARNING_RADIUS_KM = 1.0

# 需要达到的安全交会距离，单位：km
SAFE_DISTANCE_KM = 0.2

# ------------------------------------------------------------
# 轨道预测误差参数
# ------------------------------------------------------------

# 基础位置预测误差，单位：km
BASE_POSITION_ERROR_KM = 0.02

# 预测误差随预警时间增加的增长率，单位：km/h
POSITION_ERROR_GROWTH_KM_PER_HOUR = 0.0015

# ------------------------------------------------------------
# 预警时间参数
# 采用三角分布：最短时间、最可能时间、最长时间
# ------------------------------------------------------------

WARNING_TIME_MIN_HOUR = 6.0
WARNING_TIME_MODE_HOUR = 24.0
WARNING_TIME_MAX_HOUR = 72.0

# ------------------------------------------------------------
# 避撞决策参数
# ------------------------------------------------------------

# 基准碰撞概率阈值
COLLISION_PROBABILITY_THRESHOLD = 1.0e-4

# 最大可用避撞速度增量，单位：m/s
MAX_DELTA_V_MPS = 1.0

# 执行避撞后的成功率
AVOIDANCE_SUCCESS_RATE = 0.99

# 一年秒数
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0

# 输出目录
OUTPUT_DIRECTORY = Path("Q4_1_results")


# ============================================================
# 2. 基础采样函数
# ============================================================

def sample_power_law(
    rng: np.random.Generator,
    sample_size: int,
    lower: float,
    upper: float,
    beta: float,
) -> np.ndarray:
    """
    从幂律分布 f(x) = C*x^(-beta) 中抽样。

    参数：
        lower：最小值
        upper：最大值
        beta：幂律指数
    """

    if lower <= 0 or upper <= lower:
        raise ValueError("幂律分布上下限设置错误。")

    if beta <= 0:
        raise ValueError("幂律指数beta必须大于0。")

    random_values = rng.random(sample_size)

    if abs(beta - 1.0) < 1.0e-12:
        return lower * (upper / lower) ** random_values

    exponent = 1.0 - beta

    return (
        lower ** exponent
        + random_values * (upper ** exponent - lower ** exponent)
    ) ** (1.0 / exponent)


def sample_positive_normal(
    rng: np.random.Generator,
    sample_size: int,
    mean: float,
    standard_deviation: float,
) -> np.ndarray:
    """
    从正值截断正态分布中抽样。
    """

    if standard_deviation <= 0:
        raise ValueError("标准差必须大于0。")

    samples = rng.normal(mean, standard_deviation, sample_size)

    invalid_mask = samples <= 0

    while np.any(invalid_mask):
        samples[invalid_mask] = rng.normal(
            mean,
            standard_deviation,
            np.sum(invalid_mask),
        )

        invalid_mask = samples <= 0

    return samples


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """
    计算加权平均值。
    """

    weight_sum = np.sum(weights)

    if weight_sum <= 0:
        raise ValueError("权重总和必须大于0。")

    return float(np.sum(values * weights) / weight_sum)


# ============================================================
# 3. 避撞策略评价函数
# ============================================================

def evaluate_threshold(
    threshold: float,
    collision_probability: np.ndarray,
    warning_time_hour: np.ndarray,
    required_delta_v_mps: np.ndarray,
    event_weights: np.ndarray,
    annual_conjunction_rate: float,
) -> dict:
    """
    对给定碰撞概率阈值进行评价。
    """

    trigger_avoidance = (
        (collision_probability >= threshold)
        & (warning_time_hour >= WARNING_TIME_MIN_HOUR)
        & (required_delta_v_mps <= MAX_DELTA_V_MPS)
    )

    trigger_probability = weighted_mean(
        trigger_avoidance.astype(float),
        event_weights,
    )

    expected_avoidance_number = (
        annual_conjunction_rate * trigger_probability
    )

    residual_collision_probability_per_event = (
        collision_probability
        * (
            1.0
            - AVOIDANCE_SUCCESS_RATE
            * trigger_avoidance.astype(float)
        )
    )

    residual_collision_rate = (
        annual_conjunction_rate
        * weighted_mean(
            residual_collision_probability_per_event,
            event_weights,
        )
    )

    annual_residual_collision_probability = (
        -np.expm1(-residual_collision_rate)
    )

    return {
        "threshold": threshold,
        "trigger_probability": trigger_probability,
        "expected_avoidance_number": expected_avoidance_number,
        "residual_collision_rate": residual_collision_rate,
        "annual_residual_collision_probability":
            annual_residual_collision_probability,
    }


# ============================================================
# 4. 主程序
# ============================================================

def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 72)
    print("Q4(1) Single-Satellite Collision and Avoidance Model")
    print("=" * 72)
    print(f"Monte Carlo samples: {N_MONTE_CARLO:,}")
    print("Generating random conjunction events...")

    # --------------------------------------------------------
    # 4.1 碎片尺寸采样
    # --------------------------------------------------------

    debris_diameter_m = sample_power_law(
        rng=rng,
        sample_size=N_MONTE_CARLO,
        lower=DEBRIS_DIAMETER_MIN,
        upper=DEBRIS_DIAMETER_MAX,
        beta=SIZE_POWER_BETA,
    )

    # --------------------------------------------------------
    # 4.2 相对速度采样
    # --------------------------------------------------------

    relative_speed_kms = sample_positive_normal(
        rng=rng,
        sample_size=N_MONTE_CARLO,
        mean=RELATIVE_SPEED_MEAN,
        standard_deviation=RELATIVE_SPEED_STD,
    )

    # --------------------------------------------------------
    # 4.3 预警时间采样
    # --------------------------------------------------------

    warning_time_hour = rng.triangular(
        WARNING_TIME_MIN_HOUR,
        WARNING_TIME_MODE_HOUR,
        WARNING_TIME_MAX_HOUR,
        N_MONTE_CARLO,
    )

    # --------------------------------------------------------
    # 4.4 最近交会距离采样
    #
    # 假设碎片轨迹在半径为b_w的筛选圆面中均匀分布。
    # 因此距离满足：
    # F(b) = b^2 / b_w^2
    # b = b_w * sqrt(U)
    # --------------------------------------------------------

    miss_distance_km = (
        WARNING_RADIUS_KM
        * np.sqrt(rng.random(N_MONTE_CARLO))
    )

    # --------------------------------------------------------
    # 4.5 卫星与碎片硬体碰撞半径
    # --------------------------------------------------------

    satellite_equivalent_radius_km = (
        np.sqrt(SATELLITE_AREA_M2 / np.pi) / 1000.0
    )

    debris_radius_km = debris_diameter_m / 2000.0

    hard_body_radius_km = (
        satellite_equivalent_radius_km
        + debris_radius_km
    )

    # 真实硬体碰撞截面积
    collision_cross_section_km2 = (
        np.pi * hard_body_radius_km ** 2
    )

    # 危险交会筛选截面积
    warning_cross_section_km2 = (
        np.pi
        * (WARNING_RADIUS_KM + debris_radius_km) ** 2
    )

    # --------------------------------------------------------
    # 4.6 不考虑避撞时的自然碰撞率
    #
    # lambda = n*T*E[sigma*v]
    # --------------------------------------------------------

    annual_collision_rate_analytical = (
        DEBRIS_DENSITY
        * SECONDS_PER_YEAR
        * np.mean(
            collision_cross_section_km2
            * relative_speed_kms
        )
    )

    annual_collision_probability_analytical = (
        -np.expm1(-annual_collision_rate_analytical)
    )

    # --------------------------------------------------------
    # 4.7 危险交会率
    # --------------------------------------------------------

    annual_conjunction_rate = (
        DEBRIS_DENSITY
        * SECONDS_PER_YEAR
        * np.mean(
            warning_cross_section_km2
            * relative_speed_kms
        )
    )

    # 交会事件加权权重
    # 相对速度越高、筛选截面积越大，形成交会事件的概率越高
    event_weights = (
        warning_cross_section_km2
        * relative_speed_kms
    )

    # --------------------------------------------------------
    # 4.8 轨道预测误差
    # --------------------------------------------------------

    position_error_km = np.sqrt(
        BASE_POSITION_ERROR_KM ** 2
        + (
            POSITION_ERROR_GROWTH_KM_PER_HOUR
            * warning_time_hour
        ) ** 2
    )

    # --------------------------------------------------------
    # 4.9 单次交会碰撞概率
    #
    # 二维正态误差条件下的小目标近似：
    #
    # Pc = Rc^2/(2*s^2) * exp[-b^2/(2*s^2)]
    # --------------------------------------------------------

    collision_probability_event = (
        hard_body_radius_km ** 2
        / (2.0 * position_error_km ** 2)
        * np.exp(
            -miss_distance_km ** 2
            / (2.0 * position_error_km ** 2)
        )
    )

    collision_probability_event = np.clip(
        collision_probability_event,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # 4.10 所需速度增量
    #
    # Delta r ≈ Delta v * tau
    # --------------------------------------------------------

    required_displacement_km = np.maximum(
        0.0,
        SAFE_DISTANCE_KM - miss_distance_km,
    )

    required_delta_v_mps = (
        1000.0 * required_displacement_km
        / (3600.0 * warning_time_hour)
    )

    # --------------------------------------------------------
    # 4.11 Monte Carlo模型一致性验证
    #
    # 将所有交会事件的Pc加权积分，应接近硬体通量模型结果
    # --------------------------------------------------------

    annual_collision_rate_monte_carlo = (
        annual_conjunction_rate
        * weighted_mean(
            collision_probability_event,
            event_weights,
        )
    )

    annual_collision_probability_monte_carlo = (
        -np.expm1(-annual_collision_rate_monte_carlo)
    )

    consistency_relative_error = (
        abs(
            annual_collision_rate_monte_carlo
            - annual_collision_rate_analytical
        )
        / annual_collision_rate_analytical
    )

    # --------------------------------------------------------
    # 4.12 基准阈值结果
    # --------------------------------------------------------

    baseline_result = evaluate_threshold(
        threshold=COLLISION_PROBABILITY_THRESHOLD,
        collision_probability=collision_probability_event,
        warning_time_hour=warning_time_hour,
        required_delta_v_mps=required_delta_v_mps,
        event_weights=event_weights,
        annual_conjunction_rate=annual_conjunction_rate,
    )

    residual_rate = baseline_result["residual_collision_rate"]
    residual_probability = baseline_result[
        "annual_residual_collision_probability"
    ]

    risk_reduction_ratio = (
        1.0
        - residual_rate / annual_collision_rate_monte_carlo
    )

    # --------------------------------------------------------
    # 4.13 不同避撞阈值的敏感性分析
    # --------------------------------------------------------

    threshold_values = np.logspace(-7, -3, 41)

    threshold_results = []

    for threshold in threshold_values:
        result = evaluate_threshold(
            threshold=threshold,
            collision_probability=collision_probability_event,
            warning_time_hour=warning_time_hour,
            required_delta_v_mps=required_delta_v_mps,
            event_weights=event_weights,
            annual_conjunction_rate=annual_conjunction_rate,
        )

        threshold_results.append(result)

    threshold_dataframe = pd.DataFrame(threshold_results)

    threshold_csv_path = (
        OUTPUT_DIRECTORY / "Q4_1_threshold_analysis.csv"
    )

    threshold_dataframe.to_csv(
        threshold_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 4.14 绘制阈值敏感性曲线
    # --------------------------------------------------------

    figure, axis_left = plt.subplots(figsize=(9, 6))

    line_1 = axis_left.semilogx(
        threshold_dataframe["threshold"],
        threshold_dataframe["expected_avoidance_number"],
        marker="o",
        markersize=3,
        label="Expected avoidance maneuvers",
    )

    axis_left.set_xlabel("Collision probability threshold")
    axis_left.set_ylabel(
        "Expected avoidance maneuvers per satellite per year"
    )
    axis_left.grid(True, alpha=0.3)

    axis_right = axis_left.twinx()

    line_2 = axis_right.semilogx(
        threshold_dataframe["threshold"],
        threshold_dataframe[
            "annual_residual_collision_probability"
        ],
        marker="s",
        markersize=3,
        label="Residual collision probability",
    )

    axis_right.set_ylabel(
        "Annual residual collision probability"
    )

    baseline_line = axis_left.axvline(
        COLLISION_PROBABILITY_THRESHOLD,
        linestyle="--",
        label="Baseline threshold",
    )

    all_lines = line_1 + line_2 + [baseline_line]
    all_labels = [
        line.get_label()
        for line in all_lines
    ]

    axis_left.legend(
        all_lines,
        all_labels,
        loc="best",
    )

    plt.title(
        "Collision Threshold, Avoidance Frequency and Residual Risk"
    )

    plt.tight_layout()

    threshold_figure_path = (
        OUTPUT_DIRECTORY / "Q4_1_threshold_curve.png"
    )

    plt.savefig(
        threshold_figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # --------------------------------------------------------
    # 4.15 输出交会事件统计信息
    # --------------------------------------------------------

    event_statistics = pd.DataFrame(
        {
            "variable": [
                "debris_diameter_m",
                "relative_speed_km_s",
                "warning_time_hour",
                "miss_distance_km",
                "position_error_km",
                "collision_probability",
                "required_delta_v_m_s",
            ],
            "mean": [
                np.mean(debris_diameter_m),
                np.mean(relative_speed_kms),
                np.mean(warning_time_hour),
                np.mean(miss_distance_km),
                np.mean(position_error_km),
                np.mean(collision_probability_event),
                np.mean(required_delta_v_mps),
            ],
            "median": [
                np.median(debris_diameter_m),
                np.median(relative_speed_kms),
                np.median(warning_time_hour),
                np.median(miss_distance_km),
                np.median(position_error_km),
                np.median(collision_probability_event),
                np.median(required_delta_v_mps),
            ],
            "percentile_95": [
                np.percentile(debris_diameter_m, 95),
                np.percentile(relative_speed_kms, 95),
                np.percentile(warning_time_hour, 95),
                np.percentile(miss_distance_km, 95),
                np.percentile(position_error_km, 95),
                np.percentile(collision_probability_event, 95),
                np.percentile(required_delta_v_mps, 95),
            ],
            "maximum": [
                np.max(debris_diameter_m),
                np.max(relative_speed_kms),
                np.max(warning_time_hour),
                np.max(miss_distance_km),
                np.max(position_error_km),
                np.max(collision_probability_event),
                np.max(required_delta_v_mps),
            ],
        }
    )

    event_statistics_path = (
        OUTPUT_DIRECTORY / "Q4_1_event_statistics.csv"
    )

    event_statistics.to_csv(
        event_statistics_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 4.16 生成总结文件
    # --------------------------------------------------------

    summary_text = f"""
问题四第（1）问：单颗卫星碰撞概率与避撞决策模型
======================================================================

一、基准模型参数

碎片数密度：
n0 = {DEBRIS_DENSITY:.6e} 个/km^3

卫星等效迎风面积：
As = {SATELLITE_AREA_M2:.2f} m^2

碎片直径范围：
d = {DEBRIS_DIAMETER_MIN:.3f} ~ {DEBRIS_DIAMETER_MAX:.3f} m

碎片尺寸幂律指数：
beta = {SIZE_POWER_BETA:.3f}

相对速度：
均值 = {RELATIVE_SPEED_MEAN:.3f} km/s
标准差 = {RELATIVE_SPEED_STD:.3f} km/s

危险交会筛选半径：
bw = {WARNING_RADIUS_KM:.3f} km

安全距离：
bsafe = {SAFE_DISTANCE_KM:.3f} km

基准碰撞概率阈值：
Pth = {COLLISION_PROBABILITY_THRESHOLD:.6e}

最大速度增量：
Delta v_max = {MAX_DELTA_V_MPS:.3f} m/s

避撞成功率：
eta = {AVOIDANCE_SUCCESS_RATE:.4f}

Monte Carlo样本数：
N = {N_MONTE_CARLO:,}

二、自然碰撞风险

通量解析模型年均碰撞次数：
lambda_collision = {annual_collision_rate_analytical:.8e} 次/年

通量解析模型年度碰撞概率：
P_collision = {annual_collision_probability_analytical:.8e}
P_collision_percent = {100.0 * annual_collision_probability_analytical:.8f} %

Monte Carlo积分年均碰撞次数：
lambda_MC = {annual_collision_rate_monte_carlo:.8e} 次/年

Monte Carlo积分年度碰撞概率：
P_MC = {annual_collision_probability_monte_carlo:.8e}

解析模型与Monte Carlo模型相对误差：
relative_error = {100.0 * consistency_relative_error:.6f} %

三、危险交会与避撞决策

单颗卫星年均危险交会次数：
lambda_conjunction = {annual_conjunction_rate:.6f} 次/年

触发避撞的加权概率：
P_trigger = {baseline_result["trigger_probability"]:.8f}

单颗卫星年均避撞次数：
E_N_avoid = {baseline_result["expected_avoidance_number"]:.8f} 次/年

四、执行避撞后的残余风险

残余年均碰撞次数：
lambda_residual = {residual_rate:.8e} 次/年

残余年度碰撞概率：
P_residual = {residual_probability:.8e}

避撞策略风险降低比例：
risk_reduction = {100.0 * risk_reduction_ratio:.6f} %

五、模型结论

在当前参数假设下，单颗卫星自然年度碰撞概率处于较低量级，
但每年仍会出现若干次进入预警筛选区域的危险交会事件。

当碰撞概率超过阈值、预警时间充分且所需速度增量不超过
1 m/s时执行规避，可显著降低残余碰撞风险。

碰撞概率阈值越低，避撞次数越多，残余碰撞风险越小；
碰撞概率阈值越高，避撞成本越低，但残余碰撞风险增大。
因此后续可结合单次避撞成本和卫星失效损失优化阈值。
======================================================================
""".strip()

    summary_path = (
        OUTPUT_DIRECTORY / "Q4_1_summary.txt"
    )

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 4.17 终端打印
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("Calculation Results")
    print("=" * 72)

    print(
        "Analytical annual collision rate: "
        f"{annual_collision_rate_analytical:.8e} per year"
    )

    print(
        "Analytical annual collision probability: "
        f"{annual_collision_probability_analytical:.8e}"
    )

    print(
        "Monte Carlo annual collision rate: "
        f"{annual_collision_rate_monte_carlo:.8e} per year"
    )

    print(
        "Model consistency relative error: "
        f"{100.0 * consistency_relative_error:.4f}%"
    )

    print(
        "Annual conjunction rate: "
        f"{annual_conjunction_rate:.6f} per year"
    )

    print(
        "Expected avoidance maneuvers: "
        f'{baseline_result["expected_avoidance_number"]:.6f} '
        "per satellite per year"
    )

    print(
        "Residual annual collision probability: "
        f"{residual_probability:.8e}"
    )

    print(
        "Risk reduction ratio: "
        f"{100.0 * risk_reduction_ratio:.4f}%"
    )

    print()
    print("Output files:")
    print(f"1. {summary_path.resolve()}")
    print(f"2. {threshold_csv_path.resolve()}")
    print(f"3. {event_statistics_path.resolve()}")
    print(f"4. {threshold_figure_path.resolve()}")

    print("=" * 72)


if __name__ == "__main__":
    main()