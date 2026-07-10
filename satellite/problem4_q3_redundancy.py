from __future__ import annotations

"""
问题四第（3）问：星座鲁棒性与冗余配置比较

使用方法：
1. 将本文件放到 problem2_model.py 同一文件夹；
2. 安装依赖：
       python -m pip install numpy scipy matplotlib
3. 运行：
       python problem4_q3_redundancy.py

模型口径：
- 基准星座采用问题二第（2）问的单重覆盖方案：
      M=38, N=44, i=53°, F=1, 共1672颗。
- 避撞期间按保守情景处理：该卫星暂时退出稳定覆盖服务。
- 99%要求采用严格指标：
      任一颗卫星退出后，全区域仍保持至少单重覆盖的时间比例 >= 99%。
- 同时利用问题四第（1）问结果，估计全年避撞/碰撞造成的期望覆盖降级时间。
- 比较：
      A. 每个轨道面增加备用卫星；
      B. 增加额外轨道面；
      C. 设置地面备用卫星。
"""

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import poisson

from problem2_model import (
    ConstellationConfig,
    ConstellationEvaluator,
    SimulationResolution,
)


# ============================================================
# 1. 基准参数：问题二第（2）问单重覆盖方案
# ============================================================
BASE_CONFIG = ConstellationConfig(
    planes=38,
    satellites_per_plane=44,
    inclination_deg=53.0,
    phase_factor=1,
    raan_offset_deg=0.0,
    initial_phase_deg=0.0,
)

# 问题四第（1）问给出的单星风险结果
ANNUAL_AVOIDANCE_RATE_PER_SAT = 0.13611933       # 次/(颗·年)
RESIDUAL_COLLISION_PROB_PER_SAT_YEAR = 7.2072e-6 # 1/(颗·年)，小概率下近似事件率

# 问题四第（2）问采用的避撞影响时长
AVOIDANCE_DURATION_H = 6.0

# 碰撞失效后同轨卫星填补空缺耗时
COLLISION_RECOVERY_DURATION_H = 7.0 * 24.0

# 可靠性要求
TARGET_FULL_REGION_RATIO = 0.99

# 5年寿命期内地面备用星数量采用99%泊松分位数
DESIGN_LIFE_YEARS = 5.0
GROUND_SPARE_CONFIDENCE = 0.99

# 成本参数
SATELLITE_COST_YUAN = 5_000_000.0
LAUNCH_COST_YUAN = 200_000_000.0
SATELLITES_PER_LAUNCH = 60

# 搜索范围
MAX_EXTRA_SATELLITES_PER_PLANE = 2  # A方案最多测试每轨+2颗
MAX_EXTRA_PLANES = 2                # B方案最多测试+2个轨道面

# False：抽样F并在关键值附近搜索，运行更快
# True：枚举该M下全部F，结果更完整但更慢
FULL_PHASE_SEARCH = False

# 搜索阶段：先用该精度比较方案
SEARCH_RESOLUTION = SimulationResolution(
    grid_step_deg=1.0,
    time_step_s=120.0,
    duration_s=24.0 * 3600.0,
)

# 最终复核开关。第一次运行建议False。
RUN_FINAL_VALIDATION = False

# 最终论文复核精度
FINAL_RESOLUTION = SimulationResolution(
    grid_step_deg=0.5,
    time_step_s=30.0,
    duration_s=72.0 * 3600.0,
)

OUTPUT_DIR = Path("results_problem4_q3")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. 结果结构
# ============================================================
@dataclass
class RobustnessResult:
    scheme: str
    detail: str

    planes: int
    satellites_per_plane: int
    inclination_deg: float
    phase_factor: int
    total_satellites: int
    added_in_orbit_satellites: int

    grid_step_deg: float
    time_step_s: float
    duration_h: float
    ground_points: int
    time_steps: int

    baseline_full_region_ratio: float
    worst_single_outage_ratio: float
    mean_single_outage_ratio: float
    best_single_outage_ratio: float

    worst_satellite_id: int
    worst_satellite_plane: int
    worst_satellite_slot: int

    annual_avoidance_events: float
    annual_collision_events: float
    expected_degraded_hours_per_year: float
    expected_annual_full_region_ratio: float

    passes_99_percent: bool

    launch_count: int
    manufacturing_cost_yuan: float
    launch_cost_yuan: float
    total_cost_yuan: float
    incremental_cost_yuan: float

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# 3. 工具函数
# ============================================================
def construction_cost(total_satellites: int) -> tuple[int, float, float, float]:
    launch_count = math.ceil(total_satellites / SATELLITES_PER_LAUNCH)
    manufacturing = total_satellites * SATELLITE_COST_YUAN
    launch = launch_count * LAUNCH_COST_YUAN
    return launch_count, manufacturing, launch, manufacturing + launch


BASE_TOTAL_COST_YUAN = construction_cost(BASE_CONFIG.total_satellites)[3]


def phase_candidates(planes: int) -> list[int]:
    """给定轨道面数M，生成待搜索的Walker相位因子F。"""
    if FULL_PHASE_SEARCH:
        return list(range(planes))

    values = {
        0,
        1 % planes,
        2 % planes,
        (planes - 1) % planes,
        (planes - 2) % planes,
        (planes // 4) % planes,
        (planes // 2) % planes,
        (3 * planes // 4) % planes,
    }

    # 再加入若干均匀抽样值，避免只检查少量特殊点
    sample_count = min(9, planes)
    for value in np.linspace(0, planes - 1, sample_count):
        values.add(int(round(value)) % planes)

    return sorted(values)


def query_neighbours(
    tree: cKDTree,
    ground_vectors: np.ndarray,
    radius: float,
):
    """兼容不同SciPy版本。"""
    try:
        return tree.query_ball_point(
            ground_vectors,
            r=radius,
            workers=-1,
        )
    except TypeError:
        return tree.query_ball_point(
            ground_vectors,
            r=radius,
        )


def save_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 4. 单星退出鲁棒性评价
# ============================================================
def evaluate_single_satellite_outages(
    config: ConstellationConfig,
    resolution: SimulationResolution,
    scheme: str,
    detail: str,
    added_in_orbit_satellites: int,
    show_progress: bool = True,
    save_satellite_table: bool = False,
) -> RobustnessResult:
    """
    一次仿真同时计算所有卫星逐颗退出后的覆盖时间比例。

    关键逻辑：
    - 正常状态下，若某地面点覆盖重数=1，则存在唯一覆盖卫星；
    - 删除该唯一卫星后，该时刻全区域单重覆盖立即失败；
    - 因此无需将每颗卫星单独重跑一遍，只需逐时刻统计
      “哪些卫星是某些薄弱点的唯一覆盖者”。
    """
    evaluator = ConstellationEvaluator(resolution)
    evaluator._validate_config(config)
    static_arrays = evaluator._static_satellite_arrays(config)

    sat_count = config.total_satellites
    ground_count = evaluator.ground_vectors_ecef.shape[0]
    time_count = len(evaluator.times)

    # bad_steps[s]：卫星s退出时，全区域无法保持单重覆盖的时刻数
    bad_steps = np.zeros(sat_count, dtype=np.int64)
    baseline_bad_steps = 0

    for time_index, t in enumerate(evaluator.times):
        sat_vectors = evaluator._satellite_vectors_ecef(t, static_arrays)
        sat_tree = cKDTree(sat_vectors)

        neighbours = query_neighbours(
            sat_tree,
            evaluator.ground_vectors_ecef,
            evaluator.coverage_chord,
        )

        counts = np.fromiter(
            (len(item) for item in neighbours),
            dtype=np.int16,
            count=ground_count,
        )

        # 若原始方案本时刻已有未覆盖点，那么删去任意卫星都无法通过
        if np.any(counts == 0):
            baseline_bad_steps += 1
            bad_steps += 1
        else:
            unique_point_indices = np.flatnonzero(counts == 1)
            if unique_point_indices.size > 0:
                owners = np.fromiter(
                    (neighbours[index][0] for index in unique_point_indices),
                    dtype=np.int32,
                    count=unique_point_indices.size,
                )
                # 同一时刻同一卫星无论造成多少个薄弱点失效，只记1个失败时刻
                unique_owners = np.unique(owners)
                bad_steps[unique_owners] += 1

        if show_progress and (
            time_index % max(1, time_count // 20) == 0
            or time_index == time_count - 1
        ):
            percent = 100.0 * (time_index + 1) / time_count
            print(
                f"  {scheme} | M={config.planes}, N={config.satellites_per_plane}, "
                f"F={config.phase_factor}: {percent:6.2f}%"
            )

    baseline_ratio = 1.0 - baseline_bad_steps / time_count
    outage_ratios = 1.0 - bad_steps / time_count

    worst_id = int(np.argmin(outage_ratios))
    worst_plane, worst_slot = divmod(
        worst_id,
        config.satellites_per_plane,
    )

    worst_ratio = float(np.min(outage_ratios))
    mean_ratio = float(np.mean(outage_ratios))
    best_ratio = float(np.max(outage_ratios))

    # 每次单星退出期间，全区域发生覆盖降级的平均时间比例
    mean_degraded_fraction_during_outage = float(
        np.mean(1.0 - outage_ratios)
    )

    annual_avoidance_events = (
        config.total_satellites * ANNUAL_AVOIDANCE_RATE_PER_SAT
    )
    annual_collision_events = (
        config.total_satellites * RESIDUAL_COLLISION_PROB_PER_SAT_YEAR
    )

    # 假设事件稀疏、不同单星退出区间的重叠可以忽略
    expected_degraded_hours = (
        annual_avoidance_events
        * AVOIDANCE_DURATION_H
        * mean_degraded_fraction_during_outage
        +
        annual_collision_events
        * COLLISION_RECOVERY_DURATION_H
        * mean_degraded_fraction_during_outage
    )

    expected_annual_ratio = max(
        0.0,
        1.0 - expected_degraded_hours / 8760.0,
    )

    passes = bool(
        baseline_ratio >= TARGET_FULL_REGION_RATIO
        and worst_ratio >= TARGET_FULL_REGION_RATIO
        and expected_annual_ratio >= TARGET_FULL_REGION_RATIO
    )

    launch_count, manufacturing, launch, total_cost = construction_cost(
        config.total_satellites
    )

    result = RobustnessResult(
        scheme=scheme,
        detail=detail,
        planes=config.planes,
        satellites_per_plane=config.satellites_per_plane,
        inclination_deg=config.inclination_deg,
        phase_factor=config.phase_factor,
        total_satellites=config.total_satellites,
        added_in_orbit_satellites=added_in_orbit_satellites,
        grid_step_deg=resolution.grid_step_deg,
        time_step_s=resolution.time_step_s,
        duration_h=resolution.duration_s / 3600.0,
        ground_points=ground_count,
        time_steps=time_count,
        baseline_full_region_ratio=baseline_ratio,
        worst_single_outage_ratio=worst_ratio,
        mean_single_outage_ratio=mean_ratio,
        best_single_outage_ratio=best_ratio,
        worst_satellite_id=worst_id,
        worst_satellite_plane=worst_plane,
        worst_satellite_slot=worst_slot,
        annual_avoidance_events=annual_avoidance_events,
        annual_collision_events=annual_collision_events,
        expected_degraded_hours_per_year=expected_degraded_hours,
        expected_annual_full_region_ratio=expected_annual_ratio,
        passes_99_percent=passes,
        launch_count=launch_count,
        manufacturing_cost_yuan=manufacturing,
        launch_cost_yuan=launch,
        total_cost_yuan=total_cost,
        incremental_cost_yuan=total_cost - BASE_TOTAL_COST_YUAN,
    )

    if save_satellite_table:
        satellite_rows = []
        for sat_id, ratio in enumerate(outage_ratios):
            plane, slot = divmod(
                int(sat_id),
                config.satellites_per_plane,
            )
            satellite_rows.append(
                {
                    "satellite_id": int(sat_id),
                    "plane": int(plane),
                    "slot": int(slot),
                    "full_region_ratio_after_outage": float(ratio),
                    "degraded_time_ratio": float(1.0 - ratio),
                    "passes_99_percent": bool(
                        ratio >= TARGET_FULL_REGION_RATIO
                    ),
                }
            )

        safe_name = (
            f"{scheme}_M{config.planes}_N{config.satellites_per_plane}"
            f"_F{config.phase_factor}"
        ).replace(" ", "_").replace("/", "_")

        save_rows(
            OUTPUT_DIR / f"{safe_name}_single_satellite_outages.csv",
            satellite_rows,
        )

    return result


# ============================================================
# 5. 对一个M、N搜索较优F
# ============================================================
def choose_best_phase(
    planes: int,
    satellites_per_plane: int,
    scheme: str,
    detail: str,
    added_in_orbit_satellites: int,
    resolution: SimulationResolution,
) -> tuple[RobustnessResult, list[RobustnessResult]]:
    results: list[RobustnessResult] = []

    candidates = phase_candidates(planes)
    print(
        f"\n开始搜索：{scheme}，M={planes}, N={satellites_per_plane}，"
        f"待检查F={candidates}"
    )

    for phase_factor in candidates:
        config = ConstellationConfig(
            planes=planes,
            satellites_per_plane=satellites_per_plane,
            inclination_deg=53.0,
            phase_factor=phase_factor,
            raan_offset_deg=0.0,
            initial_phase_deg=0.0,
        )

        result = evaluate_single_satellite_outages(
            config=config,
            resolution=resolution,
            scheme=scheme,
            detail=detail,
            added_in_orbit_satellites=added_in_orbit_satellites,
            show_progress=False,
            save_satellite_table=False,
        )
        results.append(result)

        print(
            f"  F={phase_factor:2d} | "
            f"正常Q1={100*result.baseline_full_region_ratio:9.5f}% | "
            f"最差单星退出Q1={100*result.worst_single_outage_ratio:9.5f}% | "
            f"年期望Q1={100*result.expected_annual_full_region_ratio:9.6f}% | "
            f"通过={result.passes_99_percent}"
        )

    # 优先选择通过99%约束的方案；
    # 若均未通过，则选择最差单星退出覆盖比例最高的方案。
    results.sort(
        key=lambda item: (
            0 if item.passes_99_percent else 1,
            -item.worst_single_outage_ratio,
            -item.expected_annual_full_region_ratio,
            item.total_cost_yuan,
            item.phase_factor,
        )
    )
    return results[0], results


# ============================================================
# 6. 三类方案搜索
# ============================================================
def evaluate_baseline(
    resolution: SimulationResolution,
) -> RobustnessResult:
    print("\n" + "=" * 78)
    print("基准方案：问题二第（2）问 38×44, i=53°, F=1")
    print("=" * 78)

    return evaluate_single_satellite_outages(
        config=BASE_CONFIG,
        resolution=resolution,
        scheme="基准方案",
        detail="38×44单重覆盖星座，不增加在轨冗余",
        added_in_orbit_satellites=0,
        show_progress=True,
        save_satellite_table=True,
    )


def search_scheme_a(
    resolution: SimulationResolution,
) -> tuple[RobustnessResult | None, list[RobustnessResult]]:
    """
    A方案：每个轨道面增加备用卫星。
    备用星按在轨运行星处理，因此每轨卫星等间隔重新排布。
    """
    all_results: list[RobustnessResult] = []
    first_feasible: RobustnessResult | None = None

    for extra_per_plane in range(
        1,
        MAX_EXTRA_SATELLITES_PER_PLANE + 1,
    ):
        n_new = BASE_CONFIG.satellites_per_plane + extra_per_plane
        added = BASE_CONFIG.planes * extra_per_plane

        best, results = choose_best_phase(
            planes=BASE_CONFIG.planes,
            satellites_per_plane=n_new,
            scheme="A_每轨增加备用星",
            detail=f"每个轨道面增加{extra_per_plane}颗在轨备用星",
            added_in_orbit_satellites=added,
            resolution=resolution,
        )
        all_results.extend(results)

        if best.passes_99_percent:
            first_feasible = best
            break

    return first_feasible, all_results


def search_scheme_b(
    resolution: SimulationResolution,
) -> tuple[RobustnessResult | None, list[RobustnessResult]]:
    """
    B方案：增加额外轨道面。
    每个新增轨道面仍配置44颗卫星，所有升交点重新均匀分布。
    """
    all_results: list[RobustnessResult] = []
    first_feasible: RobustnessResult | None = None

    for extra_planes in range(1, MAX_EXTRA_PLANES + 1):
        m_new = BASE_CONFIG.planes + extra_planes
        added = extra_planes * BASE_CONFIG.satellites_per_plane

        best, results = choose_best_phase(
            planes=m_new,
            satellites_per_plane=BASE_CONFIG.satellites_per_plane,
            scheme="B_增加额外轨道面",
            detail=f"增加{extra_planes}个轨道面，每面44颗卫星",
            added_in_orbit_satellites=added,
            resolution=resolution,
        )
        all_results.extend(results)

        if best.passes_99_percent:
            first_feasible = best
            break

    return first_feasible, all_results


def ground_spare_plan(
    baseline: RobustnessResult,
) -> dict:
    """
    C方案：地面备用星。

    地面备用星不参与当前覆盖，因此不会改变短期单星退出Q1。
    数量依据5年内碰撞失效总数的99%泊松分位数确定。
    """
    expected_failures_5y = (
        BASE_CONFIG.total_satellites
        * RESIDUAL_COLLISION_PROB_PER_SAT_YEAR
        * DESIGN_LIFE_YEARS
    )

    ground_spares = int(
        poisson.ppf(
            GROUND_SPARE_CONFIDENCE,
            expected_failures_5y,
        )
    )

    initial_manufacturing_cost = ground_spares * SATELLITE_COST_YUAN

    # 一旦需要从地面补发，只要ground_spares>0，按题目成本至少需要一次发射
    contingency_launch_count = (
        math.ceil(ground_spares / SATELLITES_PER_LAUNCH)
        if ground_spares > 0
        else 0
    )
    contingency_launch_cost = (
        contingency_launch_count * LAUNCH_COST_YUAN
    )

    return {
        "scheme": "C_地面备用卫星",
        "detail": (
            f"按5年碰撞失效数的{100*GROUND_SPARE_CONFIDENCE:.1f}%"
            "泊松分位数配置"
        ),
        "ground_spares": ground_spares,
        "expected_collision_failures_5y": expected_failures_5y,
        "short_term_worst_single_outage_ratio": (
            baseline.worst_single_outage_ratio
        ),
        "short_term_passes_99_percent": bool(
            baseline.worst_single_outage_ratio
            >= TARGET_FULL_REGION_RATIO
        ),
        "initial_extra_cost_yuan": initial_manufacturing_cost,
        "contingency_launch_count": contingency_launch_count,
        "contingency_launch_cost_yuan": contingency_launch_cost,
        "activated_total_extra_cost_yuan": (
            initial_manufacturing_cost + contingency_launch_cost
        ),
        "note": (
            "地面备用星不会立即改善避撞或故障发生后的覆盖；"
            "只有在完成发射和入轨后才起作用。"
        ),
    }


# ============================================================
# 7. 结果汇总、作图与推荐
# ============================================================
def print_result(result: RobustnessResult) -> None:
    print("\n" + "-" * 78)
    print(f"方案：{result.scheme}")
    print(f"说明：{result.detail}")
    print(
        f"M={result.planes}, N={result.satellites_per_plane}, "
        f"i={result.inclination_deg:.1f}°, F={result.phase_factor}"
    )
    print(
        f"总卫星数={result.total_satellites}，"
        f"新增在轨卫星={result.added_in_orbit_satellites}"
    )
    print(
        f"正常全区域单重覆盖时间比例："
        f"{100*result.baseline_full_region_ratio:.8f}%"
    )
    print(
        f"最差单星退出后全区域单重覆盖时间比例："
        f"{100*result.worst_single_outage_ratio:.8f}%"
    )
    print(
        f"最差卫星：ID={result.worst_satellite_id}, "
        f"轨道面={result.worst_satellite_plane}, "
        f"轨内编号={result.worst_satellite_slot}"
    )
    print(
        f"全年期望覆盖降级时间："
        f"{result.expected_degraded_hours_per_year:.6f} h"
    )
    print(
        f"全年期望全区域覆盖时间比例："
        f"{100*result.expected_annual_full_region_ratio:.8f}%"
    )
    print(f"99%约束是否通过：{result.passes_99_percent}")
    print(
        f"总建设成本：{result.total_cost_yuan/1e8:.4f}亿元，"
        f"相对基准增量：{result.incremental_cost_yuan/1e8:.4f}亿元"
    )


def draw_comparison(
    baseline: RobustnessResult,
    scheme_a: RobustnessResult | None,
    scheme_b: RobustnessResult | None,
) -> None:
    candidates = [baseline]
    if scheme_a is not None:
        candidates.append(scheme_a)
    if scheme_b is not None:
        candidates.append(scheme_b)

    labels = [item.scheme for item in candidates]
    worst_ratios = [
        100.0 * item.worst_single_outage_ratio
        for item in candidates
    ]
    incremental_costs = [
        item.incremental_cost_yuan / 1e8
        for item in candidates
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, worst_ratios)
    plt.axhline(
        100.0 * TARGET_FULL_REGION_RATIO,
        linestyle="--",
        label="99% requirement",
    )
    plt.ylabel("Worst single-satellite-outage full-region coverage ratio (%)")
    plt.title("Robustness Comparison")
    plt.xticks(rotation=12)
    plt.ylim(
        max(0.0, min(worst_ratios) - 1.0),
        100.1,
    )
    plt.legend()
    for bar, value in zip(bars, worst_ratios):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}%",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "robustness_comparison.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, incremental_costs)
    plt.ylabel("Incremental construction cost (100 million yuan)")
    plt.title("Incremental Cost Comparison")
    plt.xticks(rotation=12)
    for bar, value in zip(bars, incremental_costs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "incremental_cost_comparison.png",
        dpi=300,
    )
    plt.close()


def select_recommendation(
    baseline: RobustnessResult,
    scheme_a: RobustnessResult | None,
    scheme_b: RobustnessResult | None,
    ground_plan: dict,
) -> str:
    ground_spares = ground_plan["ground_spares"]

    # 若原星座本身已经能承受任意单星退出并满足99%，
    # 则不需增加在轨星；地面备用仅承担长期补网。
    if baseline.passes_99_percent:
        return (
            "推荐方案：不额外增加在轨卫星，保留"
            f"{ground_spares}颗地面备用卫星用于5年寿命期内长期补网。"
            "理由：基准38×44星座已经通过任意单星退出的99%覆盖约束，"
            "继续增加在轨星会提高制造与发射成本。"
        )

    feasible_in_orbit = [
        item
        for item in (scheme_a, scheme_b)
        if item is not None and item.passes_99_percent
    ]

    if not feasible_in_orbit:
        return (
            "当前搜索范围内未找到满足99%约束的在轨冗余方案。"
            "请提高MAX_EXTRA_SATELLITES_PER_PLANE、MAX_EXTRA_PLANES，"
            "或将FULL_PHASE_SEARCH改为True后重新运行。"
        )

    feasible_in_orbit.sort(
        key=lambda item: (
            item.incremental_cost_yuan,
            item.added_in_orbit_satellites,
            -item.worst_single_outage_ratio,
        )
    )
    best = feasible_in_orbit[0]

    return (
        f"推荐方案：采用{best.scheme}，{best.detail}，"
        f"新增在轨卫星{best.added_in_orbit_satellites}颗，"
        f"增量建设成本约{best.incremental_cost_yuan/1e8:.2f}亿元；"
        f"同时保留{ground_spares}颗地面备用卫星用于长期补网。"
    )


# ============================================================
# 8. 主程序
# ============================================================
def run_all(
    resolution: SimulationResolution,
    stage_name: str,
) -> tuple[
    RobustnessResult,
    RobustnessResult | None,
    RobustnessResult | None,
    dict,
]:
    print("\n" + "#" * 78)
    print(f"问题四第（3）问：{stage_name}")
    print(
        f"网格={resolution.grid_step_deg}°，"
        f"时间步长={resolution.time_step_s}s，"
        f"时长={resolution.duration_s/3600.0}h"
    )
    print("#" * 78)

    baseline = evaluate_baseline(resolution)
    scheme_a, all_a = search_scheme_a(resolution)
    scheme_b, all_b = search_scheme_b(resolution)
    ground_plan = ground_spare_plan(baseline)

    # 保存所有搜索结果
    search_rows = [baseline.to_dict()]
    search_rows.extend(item.to_dict() for item in all_a)
    search_rows.extend(item.to_dict() for item in all_b)
    save_rows(
        OUTPUT_DIR / f"{stage_name}_all_phase_results.csv",
        search_rows,
    )

    # 对选出的方案重新保存逐星退出明细
    selected = [baseline]
    if scheme_a is not None:
        selected.append(scheme_a)
    if scheme_b is not None:
        selected.append(scheme_b)

    selected_detailed: list[RobustnessResult] = []
    for item in selected:
        config = ConstellationConfig(
            planes=item.planes,
            satellites_per_plane=item.satellites_per_plane,
            inclination_deg=item.inclination_deg,
            phase_factor=item.phase_factor,
            raan_offset_deg=0.0,
            initial_phase_deg=0.0,
        )
        detailed = evaluate_single_satellite_outages(
            config=config,
            resolution=resolution,
            scheme=item.scheme,
            detail=item.detail,
            added_in_orbit_satellites=item.added_in_orbit_satellites,
            show_progress=False,
            save_satellite_table=True,
        )
        selected_detailed.append(detailed)

    baseline = selected_detailed[0]
    index = 1
    if scheme_a is not None:
        scheme_a = selected_detailed[index]
        index += 1
    if scheme_b is not None:
        scheme_b = selected_detailed[index]

    comparison_rows = [baseline.to_dict()]
    if scheme_a is not None:
        comparison_rows.append(scheme_a.to_dict())
    if scheme_b is not None:
        comparison_rows.append(scheme_b.to_dict())

    save_rows(
        OUTPUT_DIR / f"{stage_name}_scheme_comparison.csv",
        comparison_rows,
    )
    save_rows(
        OUTPUT_DIR / f"{stage_name}_ground_spare_plan.csv",
        [ground_plan],
    )

    print_result(baseline)
    if scheme_a is not None:
        print_result(scheme_a)
    if scheme_b is not None:
        print_result(scheme_b)

    print("\n" + "-" * 78)
    print("C方案：地面备用卫星")
    print(
        f"5年期望碰撞失效数："
        f"{ground_plan['expected_collision_failures_5y']:.8f}"
    )
    print(f"建议地面备用星数量：{ground_plan['ground_spares']}颗")
    print(
        f"地面备用星初始制造增量："
        f"{ground_plan['initial_extra_cost_yuan']/1e8:.4f}亿元"
    )
    print(
        f"需要补发时的发射成本："
        f"{ground_plan['contingency_launch_cost_yuan']/1e8:.4f}亿元"
    )
    print(
        f"能否立即满足短期99%覆盖："
        f"{ground_plan['short_term_passes_99_percent']}"
    )

    recommendation = select_recommendation(
        baseline,
        scheme_a,
        scheme_b,
        ground_plan,
    )

    print("\n" + "=" * 78)
    print(recommendation)
    print("=" * 78)

    with (
        OUTPUT_DIR / f"{stage_name}_recommendation.txt"
    ).open("w", encoding="utf-8") as file:
        file.write(recommendation + "\n")

    draw_comparison(
        baseline,
        scheme_a,
        scheme_b,
    )

    return baseline, scheme_a, scheme_b, ground_plan


def main() -> None:
    run_all(
        resolution=SEARCH_RESOLUTION,
        stage_name="search",
    )

    if RUN_FINAL_VALIDATION:
        print(
            "\n开始72小时高精度复核。该步骤运行时间和内存占用会明显增加。"
        )
        run_all(
            resolution=FINAL_RESOLUTION,
            stage_name="final",
        )

    print(f"\n全部结果已保存到：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
