import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Q4(3) 第三步：三类冗余方案的可靠性与成本比较
#
# 已知：
# 1. 基准星座为 41×48，共 1968 颗卫星；
# 2. 第二步精细仿真表明，任意单星在完整 7 天内退出时，
#    最差全区域时间可用率仍为 99.92063492%，高于 99%；
# 3. 因此满足题设覆盖指标所需的最小额外在轨冗余数为 0。
#
# 本程序进一步比较：
# A. 每轨道面增加 1 颗在轨冗余星：41×49，增加 41 颗；
# B. 增加 1 个额外轨道面：42×48，增加 48 颗；
# C. 基准星座不增加在轨星，仅设置地面备用星。
# ============================================================


# ------------------------------------------------------------
# 1. 基础参数
# ------------------------------------------------------------
BASE_PLANES = 41
BASE_SATS_PER_PLANE = 48
BASE_SATELLITES = BASE_PLANES * BASE_SATS_PER_PLANE

DESIGN_LIFE_YEARS = 5.0

# 来自 Q4(1)
RESIDUAL_FAILURE_RATE_PER_SAT_YEAR = 7.20718274e-6
ANNUAL_AVOIDANCE_RATE_PER_SAT = 0.13611933

# 来自题目
SATELLITE_MANUFACTURING_COST_YUAN = 5_000_000.0
LAUNCH_COST_YUAN = 200_000_000.0
LAUNCH_CAPACITY = 60
AVOIDANCE_COST_PER_EVENT_YUAN = 20_000.0

# 来自 Q4(3) 第二步
WORST_SINGLE_FAILURE_AVAILABILITY = 0.9992063492
RELIABILITY_REQUIREMENT = 0.99

# 地面备用星补网发射的两种成本情景
# 1. 专门发射：每次按完整一箭计算
# 2. 拼车发射：按一箭 60 星线性分摊
DEDICATED_REPLACEMENT_LAUNCH_COST_YUAN = LAUNCH_COST_YUAN
RIDESHARE_REPLACEMENT_LAUNCH_COST_YUAN = (
    LAUNCH_COST_YUAN / LAUNCH_CAPACITY
)

OUTPUT_DIR = Path("Q4_3_step3_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. 泊松分布函数
# ------------------------------------------------------------
def poisson_pmf(k, mean_value):
    return math.exp(-mean_value) * mean_value**k / math.factorial(k)


def poisson_cdf(k, mean_value):
    return sum(poisson_pmf(j, mean_value) for j in range(k + 1))


def poisson_tail_at_least(k, mean_value):
    """P(X >= k)"""
    if k <= 0:
        return 1.0
    return 1.0 - poisson_cdf(k - 1, mean_value)


def expected_used_spares(spare_number, mean_value):
    """
    E[min(X, G)] = sum_{j=1}^{G} P(X >= j)
    X 为五年内碰撞失效数，G 为地面备用星数量。
    """
    return sum(
        poisson_tail_at_least(j, mean_value)
        for j in range(1, spare_number + 1)
    )


def minimum_ground_spares(mean_value, target_probability):
    """
    找到最小 G，使 P(X <= G) >= target_probability。
    """
    for spare_number in range(0, 20):
        if poisson_cdf(spare_number, mean_value) >= target_probability:
            return spare_number
    raise RuntimeError("备用星搜索上限过小。")


# ------------------------------------------------------------
# 3. 方案成本函数
# ------------------------------------------------------------
def initial_launch_number(extra_on_orbit_satellites):
    if extra_on_orbit_satellites <= 0:
        return 0
    return math.ceil(extra_on_orbit_satellites / LAUNCH_CAPACITY)


def evaluate_strategy(
    name,
    extra_on_orbit_satellites,
    ground_spares,
    immediate_redundancy_description,
):
    total_on_orbit = BASE_SATELLITES + extra_on_orbit_satellites

    five_year_failure_mean = (
        total_on_orbit
        * RESIDUAL_FAILURE_RATE_PER_SAT_YEAR
        * DESIGN_LIFE_YEARS
    )

    initial_manufacturing_cost = (
        extra_on_orbit_satellites + ground_spares
    ) * SATELLITE_MANUFACTURING_COST_YUAN

    launch_number = initial_launch_number(extra_on_orbit_satellites)
    initial_launch_cost = launch_number * LAUNCH_COST_YUAN

    additional_avoidance_cost = (
        extra_on_orbit_satellites
        * ANNUAL_AVOIDANCE_RATE_PER_SAT
        * AVOIDANCE_COST_PER_EVENT_YUAN
        * DESIGN_LIFE_YEARS
    )

    probability_no_failure = poisson_cdf(0, five_year_failure_mean)
    probability_at_least_one = 1.0 - probability_no_failure
    probability_at_least_two = poisson_tail_at_least(
        2,
        five_year_failure_mean,
    )

    # 将在轨冗余数量或地面备用数量视为“失效数量承受上限”的库存近似。
    # 对在轨方案，这只是数量层面的保守比较，几何覆盖能力仍以前两步仿真为准。
    redundancy_units = extra_on_orbit_satellites + ground_spares
    inventory_sufficiency_probability = poisson_cdf(
        redundancy_units,
        five_year_failure_mean,
    )

    used_ground_spares_expectation = expected_used_spares(
        ground_spares,
        five_year_failure_mean,
    )

    expected_dedicated_replacement_launch_cost = (
        used_ground_spares_expectation
        * DEDICATED_REPLACEMENT_LAUNCH_COST_YUAN
    )

    expected_rideshare_replacement_launch_cost = (
        used_ground_spares_expectation
        * RIDESHARE_REPLACEMENT_LAUNCH_COST_YUAN
    )

    expected_total_cost_dedicated = (
        initial_manufacturing_cost
        + initial_launch_cost
        + additional_avoidance_cost
        + expected_dedicated_replacement_launch_cost
    )

    expected_total_cost_rideshare = (
        initial_manufacturing_cost
        + initial_launch_cost
        + additional_avoidance_cost
        + expected_rideshare_replacement_launch_cost
    )

    return {
        "strategy": name,
        "extra_on_orbit_satellites": extra_on_orbit_satellites,
        "ground_spares": ground_spares,
        "total_on_orbit_satellites": total_on_orbit,
        "initial_launch_number": launch_number,
        "five_year_expected_failures": five_year_failure_mean,
        "probability_no_failure": probability_no_failure,
        "probability_at_least_one_failure": probability_at_least_one,
        "probability_at_least_two_failures": probability_at_least_two,
        "inventory_sufficiency_probability": (
            inventory_sufficiency_probability
        ),
        "expected_used_ground_spares": used_ground_spares_expectation,
        "initial_manufacturing_cost_yuan": initial_manufacturing_cost,
        "initial_launch_cost_yuan": initial_launch_cost,
        "five_year_additional_avoidance_cost_yuan": (
            additional_avoidance_cost
        ),
        "expected_replacement_launch_cost_dedicated_yuan": (
            expected_dedicated_replacement_launch_cost
        ),
        "expected_replacement_launch_cost_rideshare_yuan": (
            expected_rideshare_replacement_launch_cost
        ),
        "expected_total_cost_dedicated_yuan": (
            expected_total_cost_dedicated
        ),
        "expected_total_cost_rideshare_yuan": (
            expected_total_cost_rideshare
        ),
        "immediate_redundancy_description": (
            immediate_redundancy_description
        ),
    }


# ------------------------------------------------------------
# 4. 生成方案
# ------------------------------------------------------------
def build_strategies():
    return [
        evaluate_strategy(
            name="基准方案：不额外配置",
            extra_on_orbit_satellites=0,
            ground_spares=0,
            immediate_redundancy_description=(
                "依靠原星座本身的单星退出鲁棒性"
            ),
        ),
        evaluate_strategy(
            name="方案一：每轨道面增加1颗在轨星",
            extra_on_orbit_satellites=BASE_PLANES,
            ground_spares=0,
            immediate_redundancy_description=(
                "41颗冗余星均匀分散于各轨道面，实时响应能力强"
            ),
        ),
        evaluate_strategy(
            name="方案二：增加1个额外轨道面",
            extra_on_orbit_satellites=BASE_SATS_PER_PLANE,
            ground_spares=0,
            immediate_redundancy_description=(
                "增加48颗卫星，强化跨轨覆盖与链路冗余"
            ),
        ),
        evaluate_strategy(
            name="方案三：1颗地面备用星",
            extra_on_orbit_satellites=0,
            ground_spares=1,
            immediate_redundancy_description=(
                "不能立即补位，但基准星座可承受单星退出"
            ),
        ),
        evaluate_strategy(
            name="扩展方案：2颗地面备用星",
            extra_on_orbit_satellites=0,
            ground_spares=2,
            immediate_redundancy_description=(
                "适用于更高置信度或考虑额外非碰撞故障"
            ),
        ),
    ]


# ------------------------------------------------------------
# 5. 保存结果
# ------------------------------------------------------------
def save_csv(strategies):
    path = OUTPUT_DIR / "Q4_3_step3_strategy_comparison.csv"

    fieldnames = list(strategies[0].keys())

    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(strategies)

    return path


def save_ground_spare_table(base_failure_mean):
    path = OUTPUT_DIR / "Q4_3_step3_ground_spare_reliability.csv"

    rows = []

    for ground_spares in range(0, 6):
        rows.append({
            "ground_spares": ground_spares,
            "probability_failures_not_exceed_spares": poisson_cdf(
                ground_spares,
                base_failure_mean,
            ),
            "shortage_probability": 1.0 - poisson_cdf(
                ground_spares,
                base_failure_mean,
            ),
            "expected_used_spares": expected_used_spares(
                ground_spares,
                base_failure_mean,
            ),
        })

    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    return path, rows


def save_summary(strategies, base_failure_mean):
    path = OUTPUT_DIR / "Q4_3_step3_summary.txt"

    minimum_99 = minimum_ground_spares(base_failure_mean, 0.99)
    minimum_999 = minimum_ground_spares(base_failure_mean, 0.999)

    ground_one = next(
        item
        for item in strategies
        if item["strategy"] == "方案三：1颗地面备用星"
    )
    plane_spares = next(
        item
        for item in strategies
        if item["strategy"] == "方案一：每轨道面增加1颗在轨星"
    )
    extra_plane = next(
        item
        for item in strategies
        if item["strategy"] == "方案二：增加1个额外轨道面"
    )

    text = f"""Q4(3) 第三步：冗余方案可靠性与成本比较
{'=' * 78}

一、已有覆盖鲁棒性结论
基准构型：41×48，共 {BASE_SATELLITES} 颗卫星
单星完整退出 7 天时的最差全区域时间可用率：
{WORST_SINGLE_FAILURE_AVAILABILITY * 100:.8f} %

可靠性要求：
{RELIABILITY_REQUIREMENT * 100:.2f} %

因此，在当前模型与精度下：
满足 99% 覆盖时间要求所需的最小额外在轨冗余卫星数 = 0

二、五年碰撞失效统计
单星残余年均失效率：
{RESIDUAL_FAILURE_RATE_PER_SAT_YEAR:.10e} 次/(颗·年)

基准星座五年期望碰撞失效数：
mu_5 = {base_failure_mean:.10f} 次

五年内至少发生 1 次碰撞失效的概率：
{(1.0 - poisson_cdf(0, base_failure_mean)) * 100:.8f} %

五年内至少发生 2 次碰撞失效的概率：
{poisson_tail_at_least(2, base_failure_mean) * 100:.8f} %

五年内失效数不超过 1 次的概率：
{poisson_cdf(1, base_failure_mean) * 100:.8f} %

五年内失效数不超过 2 次的概率：
{poisson_cdf(2, base_failure_mean) * 100:.8f} %

达到 99% 备用数量充足概率所需的最少地面备用星：
{minimum_99} 颗

达到 99.9% 备用数量充足概率所需的最少地面备用星：
{minimum_999} 颗

三、三种主要方案的五年增量成本

1. 每轨道面增加 1 颗在轨星
增加卫星数 = {plane_spares['extra_on_orbit_satellites']} 颗
初始制造成本 = {plane_spares['initial_manufacturing_cost_yuan'] / 1e8:.6f} 亿元
初始发射成本 = {plane_spares['initial_launch_cost_yuan'] / 1e8:.6f} 亿元
五年新增避撞成本 = {plane_spares['five_year_additional_avoidance_cost_yuan'] / 1e8:.6f} 亿元
五年增量总成本 = {plane_spares['expected_total_cost_dedicated_yuan'] / 1e8:.6f} 亿元

2. 增加 1 个额外轨道面
增加卫星数 = {extra_plane['extra_on_orbit_satellites']} 颗
初始制造成本 = {extra_plane['initial_manufacturing_cost_yuan'] / 1e8:.6f} 亿元
初始发射成本 = {extra_plane['initial_launch_cost_yuan'] / 1e8:.6f} 亿元
五年新增避撞成本 = {extra_plane['five_year_additional_avoidance_cost_yuan'] / 1e8:.6f} 亿元
五年增量总成本 = {extra_plane['expected_total_cost_dedicated_yuan'] / 1e8:.6f} 亿元

3. 设置 1 颗地面备用星
备用星制造成本 = {ground_one['initial_manufacturing_cost_yuan'] / 1e8:.6f} 亿元
期望使用地面备用星数 = {ground_one['expected_used_ground_spares']:.8f} 颗

按专门发射计算：
期望补网发射成本 = {ground_one['expected_replacement_launch_cost_dedicated_yuan'] / 1e8:.6f} 亿元
五年期望增量总成本 = {ground_one['expected_total_cost_dedicated_yuan'] / 1e8:.6f} 亿元

按拼车线性分摊计算：
期望补网发射成本 = {ground_one['expected_replacement_launch_cost_rideshare_yuan'] / 1e8:.6f} 亿元
五年期望增量总成本 = {ground_one['expected_total_cost_rideshare_yuan'] / 1e8:.6f} 亿元

四、推荐结论
1. 基准星座已经满足任意单星完整退出 7 天时的 99% 覆盖要求，
   因而不需要为满足该指标额外增加在轨卫星。
2. 按当前碰撞残余风险，1 颗地面备用星可使五年内备用数量充足概率达到
   {poisson_cdf(1, base_failure_mean) * 100:.6f}%，高于 99%。
3. “每轨增加 1 星”和“增加 1 个轨道面”都能增强实时冗余，
   但五年增量成本约为 4 亿元，远高于地面备用方案。
4. 因此当前模型下的最优配置为：
   0 颗额外在轨冗余星 + 1 颗地面备用星。
5. 若要求 99.9% 的备用数量充足概率，或需要计入电子故障、推进系统故障等
   非碰撞失效，则可将地面备用星提高到 2 颗。

{'=' * 78}
"""

    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------
# 6. 绘图
# ------------------------------------------------------------
def save_figures(strategies, ground_rows):
    # 图 1：三种主要方案的成本
    selected = [
        strategies[1],
        strategies[2],
        strategies[3],
    ]

    labels = [
        "One spare\nper plane",
        "One extra\norbital plane",
        "One ground\nspare",
    ]

    dedicated_costs = np.array([
        item["expected_total_cost_dedicated_yuan"] / 1e8
        for item in selected
    ])

    rideshare_costs = np.array([
        item["expected_total_cost_rideshare_yuan"] / 1e8
        for item in selected
    ])

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        x - width / 2,
        dedicated_costs,
        width,
        label="Dedicated replacement launch",
    )
    ax.bar(
        x + width / 2,
        rideshare_costs,
        width,
        label="Rideshare replacement launch",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Five-year incremental cost / 100 million CNY")
    ax.set_title("Cost Comparison of Redundancy Strategies")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "Q4_3_step3_cost_comparison.png",
        dpi=220,
    )

    # 图 2：地面备用星数量与备用充足概率
    spare_numbers = np.array([
        row["ground_spares"]
        for row in ground_rows
    ])
    adequacy = np.array([
        row["probability_failures_not_exceed_spares"] * 100.0
        for row in ground_rows
    ])

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.plot(spare_numbers, adequacy, marker="o")
    ax.axhline(
        99.0,
        linestyle="--",
        label="99% target",
    )
    ax.axhline(
        99.9,
        linestyle=":",
        label="99.9% target",
    )
    ax.set_xlabel("Number of ground spare satellites")
    ax.set_ylabel("Probability of sufficient spares (%)")
    ax.set_title("Ground-Spare Sufficiency over Five Years")
    ax.set_xticks(spare_numbers)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "Q4_3_step3_ground_spare_reliability.png",
        dpi=220,
    )

    plt.show()


# ------------------------------------------------------------
# 7. 主程序
# ------------------------------------------------------------
def main():
    strategies = build_strategies()

    base_failure_mean = (
        BASE_SATELLITES
        * RESIDUAL_FAILURE_RATE_PER_SAT_YEAR
        * DESIGN_LIFE_YEARS
    )

    strategy_csv = save_csv(strategies)
    spare_csv, ground_rows = save_ground_spare_table(
        base_failure_mean
    )
    summary_path = save_summary(
        strategies,
        base_failure_mean,
    )

    save_figures(strategies, ground_rows)

    ground_one = strategies[3]

    print("=" * 78)
    print("Q4(3) 第三步计算完成")
    print("=" * 78)
    print(
        "基准星座单星退出7天最差可用率 = "
        f"{WORST_SINGLE_FAILURE_AVAILABILITY * 100:.8f} %"
    )
    print("满足99%覆盖要求的最小额外在轨冗余数 = 0")
    print(
        "1颗地面备用星的五年备用充足概率 = "
        f"{ground_one['inventory_sufficiency_probability'] * 100:.8f} %"
    )
    print(
        "1颗地面备用星五年期望增量成本（专门发射） = "
        f"{ground_one['expected_total_cost_dedicated_yuan'] / 1e8:.8f} 亿元"
    )
    print(
        "1颗地面备用星五年期望增量成本（拼车分摊） = "
        f"{ground_one['expected_total_cost_rideshare_yuan'] / 1e8:.8f} 亿元"
    )
    print("\n推荐：0颗额外在轨冗余星 + 1颗地面备用星")
    print(f"\n方案比较表：{strategy_csv}")
    print(f"地面备用表：{spare_csv}")
    print(f"总结文件：{summary_path}")
    print(f"图片目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()