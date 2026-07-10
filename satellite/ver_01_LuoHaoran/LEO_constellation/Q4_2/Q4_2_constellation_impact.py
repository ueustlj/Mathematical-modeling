import csv
import os

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 问题四（2）：全星座年度避撞次数、容量损失与经济成本
#
# 基于问题二的星座方案和问题四（1）的单星避撞频率，计算：
# 1. 全星座年度平均避撞总次数；
# 2. 避撞期间通信能力下降造成的容量损失；
# 3. 年度及设计寿命内的避撞直接成本；
# 4. 避撞持续时间的敏感性；
# 5. 年度避撞次数的泊松随机波动区间。
# ============================================================


# ------------------------------------------------------------
# 1. 可修改参数
# ------------------------------------------------------------

# 问题二最终采用的二重覆盖星座方案
M = 41                          # 轨道面数
N = 48                          # 每轨道面卫星数
I_DEG = 53.0                    # 轨道倾角，仅用于结果记录
F = 1                           # Walker 相位参数，仅用于结果记录

# 问题四（1）输出：单颗卫星年均避撞次数
LAMBDA_AVOID_SINGLE = 0.13611933     # 次/(颗·年)

# 题目给定参数
CAPACITY_PER_SAT_GBPS = 20.0         # 单星接入容量，Gbps
CAPACITY_DROP_RATIO = 0.50           # 避撞期间容量下降比例
COST_PER_AVOIDANCE_YUAN = 20000.0    # 单次避撞燃料成本，元
DESIGN_LIFE_YEARS = 5                # 设计寿命，年

# 题目未给出单次避撞影响时长，因此设置基准值并做敏感性分析
BASE_DURATION_H = 6.0
DURATION_GRID_H = np.array(
    [1.0, 3.0, 6.0, 12.0, 24.0],
    dtype=float
)

# 泊松 Monte Carlo：模拟很多个“运行年份”
MONTE_CARLO_YEARS = 200_000
RANDOM_SEED = 20260710

# 是否在程序结束时弹出图片
SHOW_FIGURES = True

OUTPUT_DIR = "Q4_2_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. 派生量
# ------------------------------------------------------------

HOURS_PER_YEAR = 365.0 * 24.0
TOTAL_SATS = M * N

EXPECTED_ANNUAL_AVOIDANCES = (
    TOTAL_SATS * LAMBDA_AVOID_SINGLE
)

ANNUAL_COUNT_STD = np.sqrt(
    EXPECTED_ANNUAL_AVOIDANCES
)

TOTAL_CAPACITY_GBPS = (
    TOTAL_SATS * CAPACITY_PER_SAT_GBPS
)

DROP_PER_EVENT_GBPS = (
    CAPACITY_PER_SAT_GBPS
    * CAPACITY_DROP_RATIO
)

ANNUAL_COST_YUAN = (
    EXPECTED_ANNUAL_AVOIDANCES
    * COST_PER_AVOIDANCE_YUAN
)

LIFETIME_COST_YUAN = (
    ANNUAL_COST_YUAN
    * DESIGN_LIFE_YEARS
)

MEAN_DAYS_BETWEEN_EVENTS = (
    365.0 / EXPECTED_ANNUAL_AVOIDANCES
)


# ------------------------------------------------------------
# 3. 避撞持续时间敏感性
# ------------------------------------------------------------

def calculate_duration_sensitivity():
    rows = []

    for duration_h in DURATION_GRID_H:
        # 每次避撞造成的容量—时间损失
        single_event_loss_gbps_h = (
            DROP_PER_EVENT_GBPS
            * duration_h
        )

        # 全星座全年容量—时间损失
        annual_loss_gbps_h = (
            EXPECTED_ANNUAL_AVOIDANCES
            * single_event_loss_gbps_h
        )

        # 星座正常情况下全年可提供的总容量—时间
        annual_total_gbps_h = (
            TOTAL_CAPACITY_GBPS
            * HOURS_PER_YEAR
        )

        loss_fraction = (
            annual_loss_gbps_h
            / annual_total_gbps_h
        )

        average_capacity_loss_gbps = (
            annual_loss_gbps_h
            / HOURS_PER_YEAR
        )

        average_available_capacity_gbps = (
            TOTAL_CAPACITY_GBPS
            - average_capacity_loss_gbps
        )

        # 平均同时处于避撞影响状态的卫星数
        average_degraded_satellites = (
            EXPECTED_ANNUAL_AVOIDANCES
            * duration_h
            / HOURS_PER_YEAR
        )

        # 折算为“完全失去容量”的等效卫星数
        equivalent_full_satellite_loss = (
            average_degraded_satellites
            * CAPACITY_DROP_RATIO
        )

        rows.append({
            "duration_h": duration_h,
            "single_event_loss_gbps_h": single_event_loss_gbps_h,
            "annual_loss_gbps_h": annual_loss_gbps_h,
            "capacity_loss_fraction": loss_fraction,
            "capacity_loss_percent": 100.0 * loss_fraction,
            "average_capacity_loss_gbps": average_capacity_loss_gbps,
            "average_available_capacity_gbps": average_available_capacity_gbps,
            "average_degraded_satellites": average_degraded_satellites,
            "equivalent_full_satellite_loss": equivalent_full_satellite_loss,
        })

    return rows


# ------------------------------------------------------------
# 4. 年度避撞次数的泊松模拟
# ------------------------------------------------------------

def simulate_annual_counts():
    rng = np.random.default_rng(RANDOM_SEED)

    counts = rng.poisson(
        EXPECTED_ANNUAL_AVOIDANCES,
        size=MONTE_CARLO_YEARS
    )

    q025, q50, q975 = np.quantile(
        counts,
        [0.025, 0.50, 0.975]
    )

    return {
        "counts": counts,
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts, ddof=1)),
        "q025": int(q025),
        "q50": int(q50),
        "q975": int(q975),
    }


# ------------------------------------------------------------
# 5. 保存 CSV
# ------------------------------------------------------------

def save_duration_csv(rows):
    path = os.path.join(
        OUTPUT_DIR,
        "Q4_2_duration_sensitivity.csv"
    )

    fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def save_count_distribution_csv(counts):
    values, frequencies = np.unique(
        counts,
        return_counts=True
    )

    probabilities = (
        frequencies / counts.size
    )

    path = os.path.join(
        OUTPUT_DIR,
        "Q4_2_annual_count_distribution.csv"
    )

    data = np.column_stack((
        values,
        frequencies,
        probabilities
    ))

    np.savetxt(
        path,
        data,
        delimiter=",",
        header=(
            "annual_avoidance_count,"
            "simulation_frequency,"
            "estimated_probability"
        ),
        comments="",
        fmt=["%d", "%d", "%.10f"]
    )

    return path


# ------------------------------------------------------------
# 6. 绘图
# ------------------------------------------------------------

def plot_capacity_sensitivity(rows):
    duration = np.array([
        row["duration_h"]
        for row in rows
    ])

    loss_percent = np.array([
        row["capacity_loss_percent"]
        for row in rows
    ])

    average_loss = np.array([
        row["average_capacity_loss_gbps"]
        for row in rows
    ])

    plt.figure(figsize=(8.5, 5.5))
    plt.plot(
        duration,
        loss_percent,
        marker="o"
    )
    plt.xlabel("Avoidance impact duration / h")
    plt.ylabel("Annual capacity loss / %")
    plt.title("Capacity Loss Sensitivity to Avoidance Duration")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "Q4_2_capacity_loss_percent.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.figure(figsize=(8.5, 5.5))
    plt.plot(
        duration,
        average_loss,
        marker="o"
    )
    plt.xlabel("Avoidance impact duration / h")
    plt.ylabel("Average capacity loss / Gbps")
    plt.title("Average Constellation Capacity Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "Q4_2_average_capacity_loss.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


def plot_annual_count_distribution(simulation):
    counts = simulation["counts"]
    values, frequencies = np.unique(
        counts,
        return_counts=True
    )

    probabilities = (
        frequencies / counts.size
    )

    lower = simulation["q025"] - 8
    upper = simulation["q975"] + 8
    mask = (
        (values >= lower)
        & (values <= upper)
    )

    plt.figure(figsize=(9.0, 5.5))
    plt.bar(
        values[mask],
        probabilities[mask],
        width=0.9
    )
    plt.axvline(
        EXPECTED_ANNUAL_AVOIDANCES,
        linestyle="--",
        label="Expected value"
    )
    plt.axvline(
        simulation["q025"],
        linestyle=":",
        label="2.5% quantile"
    )
    plt.axvline(
        simulation["q975"],
        linestyle=":",
        label="97.5% quantile"
    )
    plt.xlabel("Annual constellation avoidance count")
    plt.ylabel("Estimated probability")
    plt.title("Poisson Distribution of Annual Avoidance Count")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "Q4_2_annual_avoidance_distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


# ------------------------------------------------------------
# 7. 保存总结文本
# ------------------------------------------------------------

def find_baseline_row(rows):
    for row in rows:
        if np.isclose(
            row["duration_h"],
            BASE_DURATION_H
        ):
            return row

    raise ValueError(
        "BASE_DURATION_H 必须包含在 DURATION_GRID_H 中。"
    )


def save_summary(rows, simulation):
    baseline = find_baseline_row(rows)

    annual_cost_low = (
        simulation["q025"]
        * COST_PER_AVOIDANCE_YUAN
    )

    annual_cost_high = (
        simulation["q975"]
        * COST_PER_AVOIDANCE_YUAN
    )

    text = f"""问题四第（2）问：全星座避撞影响评估
{'=' * 72}

一、星座参数
M = {M}
N = {N}
i = {I_DEG:.1f} deg
F = {F}
总卫星数 = {TOTAL_SATS} 颗

二、单星输入参数
单颗卫星年均避撞次数 = {LAMBDA_AVOID_SINGLE:.8f} 次/(颗·年)
单星接入容量 = {CAPACITY_PER_SAT_GBPS:.2f} Gbps
避撞期间容量下降比例 = {100.0 * CAPACITY_DROP_RATIO:.2f} %
单次避撞成本 = {COST_PER_AVOIDANCE_YUAN:.2f} 元

三、全星座年度避撞次数
理论期望 = {EXPECTED_ANNUAL_AVOIDANCES:.8f} 次/年
理论标准差 = {ANNUAL_COUNT_STD:.8f} 次/年
Monte Carlo均值 = {simulation['mean']:.8f} 次/年
Monte Carlo标准差 = {simulation['std']:.8f} 次/年
Monte Carlo中位数 = {simulation['q50']} 次/年
95%经验区间 = [{simulation['q025']}, {simulation['q975']}] 次/年
全星座平均每 {MEAN_DAYS_BETWEEN_EVENTS:.8f} 天发生一次避撞

四、容量损失：基准影响时长 {BASE_DURATION_H:.2f} h
星座正常总容量 = {TOTAL_CAPACITY_GBPS:.8f} Gbps
单次避撞容量下降 = {DROP_PER_EVENT_GBPS:.8f} Gbps
年度容量—时间损失 = {baseline['annual_loss_gbps_h']:.8f} Gbps·h
平均容量下降 = {baseline['average_capacity_loss_gbps']:.8f} Gbps
平均可用容量 = {baseline['average_available_capacity_gbps']:.8f} Gbps
容量下降比例 = {baseline['capacity_loss_percent']:.10f} %
平均同时受避撞影响的卫星数 = {baseline['average_degraded_satellites']:.8f} 颗
等效完全容量损失卫星数 = {baseline['equivalent_full_satellite_loss']:.8f} 颗

五、经济成本
年度平均避撞直接成本 = {ANNUAL_COST_YUAN:.8f} 元
年度平均避撞直接成本 = {ANNUAL_COST_YUAN / 10000.0:.8f} 万元
年度成本95%经验区间 = [{annual_cost_low / 10000.0:.4f}, {annual_cost_high / 10000.0:.4f}] 万元
{DESIGN_LIFE_YEARS}年累计避撞直接成本 = {LIFETIME_COST_YUAN:.8f} 元
{DESIGN_LIFE_YEARS}年累计避撞直接成本 = {LIFETIME_COST_YUAN / 10000.0:.8f} 万元

六、结论
1. 全星座年均执行约 {EXPECTED_ANNUAL_AVOIDANCES:.2f} 次避撞机动。
2. 在单次影响 {BASE_DURATION_H:.1f} h 的基准情景下，平均容量下降约
   {baseline['average_capacity_loss_gbps']:.4f} Gbps，占总容量
   {baseline['capacity_loss_percent']:.6f} %。
3. 年度避撞直接成本约 {ANNUAL_COST_YUAN / 10000.0:.2f} 万元，
   {DESIGN_LIFE_YEARS} 年累计约 {LIFETIME_COST_YUAN / 10000.0:.2f} 万元。
4. 全星座平均容量损失较小，但局部覆盖和局部拥塞风险不能由平均指标排除，
   需要在问题四第（3）问中通过冗余配置进一步分析。
{'=' * 72}
"""

    path = os.path.join(
        OUTPUT_DIR,
        "Q4_2_summary.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(text)

    return path, text


# ------------------------------------------------------------
# 8. 主程序
# ------------------------------------------------------------

def main():
    rows = calculate_duration_sensitivity()
    simulation = simulate_annual_counts()

    duration_csv = save_duration_csv(rows)
    distribution_csv = save_count_distribution_csv(
        simulation["counts"]
    )

    plot_capacity_sensitivity(rows)
    plot_annual_count_distribution(simulation)

    summary_path, summary_text = save_summary(
        rows,
        simulation
    )

    print(summary_text)
    print("输出文件已保存：")
    print(duration_csv)
    print(distribution_csv)
    print(summary_path)
    print(
        os.path.join(
            OUTPUT_DIR,
            "Q4_2_capacity_loss_percent.png"
        )
    )
    print(
        os.path.join(
            OUTPUT_DIR,
            "Q4_2_average_capacity_loss.png"
        )
    )
    print(
        os.path.join(
            OUTPUT_DIR,
            "Q4_2_annual_avoidance_distribution.png"
        )
    )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()