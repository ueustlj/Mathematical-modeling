import os

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
except ImportError as exc:
    raise SystemExit(
        "缺少 scipy。请在 VS Code 终端运行：py -m pip install scipy"
    ) from exc


# ============================================================
# 问题三（2）：区域任意两点之间的最小时延路由
#
# 模型包括：
# 1. 地面点 -> 可见卫星的上行传播时延；
# 2. 星间激光链路传播时延；
# 3. 每经过一条星间链路增加 0.5 ms 星上处理时延；
# 4. 最后一颗卫星 -> 地面点的下行传播时延；
# 5. 使用 Dijkstra 算法求端到端最小时延。
#
# 距离单位：km；时间单位：s；最终时延单位：ms。
# ============================================================


# -------------------------
# 1. 星座与物理参数
# -------------------------
R_EARTH = 6371.0
H_ORBIT = 550.0
MU = 398600.4418
OMEGA_EARTH = 7.2921159e-5
C_LIGHT = 299792.458                    # km/s

# 当前采用你上传代码中的问题二方案：41 × 48, i=53°, F=1。
# 若要验证问题二（2）的 36 × 44 方案，只需改 M 和 N。
M = 41
N = 48
I_DEG = 53.0
F = 1

MAX_LINK_DISTANCE = 5000.0             # km
COVERAGE_RADIUS_KM = 506.0             # 题目给出的地面覆盖半径
PROCESSING_DELAY_MS = 0.5              # 每条星间跳的处理时延
DELAY_LIMIT_MS = 30.0


# -------------------------
# 2. 区域与仿真精度参数
# -------------------------
LAT_MIN = 4.0
LAT_MAX = 53.0
LON_MIN = 73.0
LON_MAX = 135.0

# 正式计算采用 5° 网格、49 个时刻；试运行可临时改为 7°、25 个时刻。
GROUND_GRID_STEP_DEG = 5.0
ROUTING_TIME_STEPS = 49

SHOW_FIGURES = False
OUTPUT_DIR = "Q3_2_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# 3. 派生参数
# -------------------------
A = R_EARTH + H_ORBIT
I = np.radians(I_DEG)
OMEGA = np.sqrt(MU / A**3)
PERIOD = 2 * np.pi / OMEGA
COVERAGE_ANGLE_RAD = COVERAGE_RADIUS_KM / R_EARTH
TOTAL_SATS = M * N


def node_id(m, k):
    """把轨道面编号 m、面内编号 k 转换为图节点编号。"""
    return m * N + k


def satellite_positions(t):
    """返回时刻 t 的全部卫星 ECI 坐标，形状为 (M, N, 3)。"""
    m = np.arange(M)[:, None]
    k = np.arange(N)[None, :]

    raan = 2 * np.pi * m / M
    u = (
        OMEGA * t
        + 2 * np.pi * k / N
        + 2 * np.pi * F * m / (M * N)
    )

    x0 = A * np.cos(u)
    y0 = A * np.sin(u)

    x1 = x0
    y1 = y0 * np.cos(I)
    z1 = y0 * np.sin(I)

    x = x1 * np.cos(raan) - y1 * np.sin(raan)
    y = x1 * np.sin(raan) + y1 * np.cos(raan)

    # 原上传代码中 z1 的形状是 (1, N)，必须扩展到 (M, N)。
    z = np.broadcast_to(z1, (M, N))

    return np.stack((x, y, z), axis=-1)


def build_intra_links(pos):
    """同轨道面前后相邻星之间的无向链路。"""
    edges = []

    for m in range(M):
        for k in range(N):
            k2 = (k + 1) % N
            u = node_id(m, k)
            v = node_id(m, k2)
            d = float(np.linalg.norm(pos[m, k] - pos[m, k2]))
            edges.append((u, v, d))

    return edges


def build_inter_links(pos):
    """每颗卫星与右侧相邻轨道面内最近卫星建立跨轨链路。"""
    edges = []

    for m in range(M):
        mr = (m + 1) % M

        diff = pos[m][:, None, :] - pos[mr][None, :, :]
        distance_matrix = np.linalg.norm(diff, axis=2)
        nearest = np.argmin(distance_matrix, axis=1)

        for k in range(N):
            j = int(nearest[k])
            u = node_id(m, k)
            v = node_id(mr, j)
            d = float(distance_matrix[k, j])
            edges.append((u, v, d))

    return edges


def inclusive_axis(v_min, v_max, step):
    """生成包含上下边界的坐标轴。"""
    values = list(np.arange(v_min, v_max + 1e-12, step))

    if values[-1] < v_max - 1e-9:
        values.append(v_max)

    return np.asarray(values, dtype=float)


def make_ground_grid():
    """
    联合使用5°网格和7°网格。

    原因：
    5°网格不包含11°N，而此前最坏点恰好出现在11°N。
    将两组网格取并集，可以保留精细网格，同时避免遗漏此前
    已发现的较差点。
    """
    point_set = set()

    for step in (5.0, 7.0):
        latitudes = inclusive_axis(
            LAT_MIN,
            LAT_MAX,
            step
        )

        longitudes = inclusive_axis(
            LON_MIN,
            LON_MAX,
            step
        )

        for lat in latitudes:
            for lon in longitudes:
                point_set.add((
                    round(float(lat), 10),
                    round(float(lon), 10)
                ))

    points = sorted(
        point_set,
        key=lambda item: (item[0], item[1])
    )

    return np.asarray(points, dtype=float)


def ground_positions_eci(points, t):
    """把固定地面点转换为时刻 t 的 ECI 坐标，包含地球自转。"""
    lat = np.radians(points[:, 0])
    lon = np.radians(points[:, 1]) + OMEGA_EARTH * t

    cos_lat = np.cos(lat)

    return R_EARTH * np.column_stack((
        cos_lat * np.cos(lon),
        cos_lat * np.sin(lon),
        np.sin(lat),
    ))


def calculate_access_geometry(points, t, sat_flat):
    """
    计算每个地面点与每颗卫星的：
    1. 可见性矩阵 visible，形状为 (P, TOTAL_SATS)；
    2. 斜距传播时延 access_ms，单位 ms。
    """
    ground_pos = ground_positions_eci(points, t)

    ground_unit = ground_pos / R_EARTH
    satellite_unit = sat_flat / A
    cos_central_angle = ground_unit @ satellite_unit.T

    visible = (
        cos_central_angle
        >= np.cos(COVERAGE_ANGLE_RAD) - 1e-12
    )

    slant_distance = np.linalg.norm(
        ground_pos[:, None, :] - sat_flat[None, :, :],
        axis=2,
    )

    access_ms = slant_distance / C_LIGHT * 1000.0

    return visible, access_ms


def build_routing_graph(intra_edges, inter_edges, visible, access_ms):
    """
    构造有向稀疏图：

    satellite nodes: 0 ... TOTAL_SATS-1
    source ground nodes: TOTAL_SATS ... TOTAL_SATS+P-1
    destination ground nodes: TOTAL_SATS+P ... TOTAL_SATS+2P-1

    地面源节点只允许上行，地面目的节点只允许下行，避免路径借助
    第三个地面站产生不符合题意的“地面中继”。
    """
    point_count = visible.shape[0]
    graph_size = TOTAL_SATS + 2 * point_count

    rows = []
    cols = []
    weights = []

    # 星间链路为双向链路。
    # 从卫星 u 到卫星 v：传播时延 + 到达 v 后的 0.5 ms 处理时延。
    for u, v, distance in intra_edges + inter_edges:
        if distance <= MAX_LINK_DISTANCE:
            weight = (
                distance / C_LIGHT * 1000.0
                + PROCESSING_DELAY_MS
            )

            rows.extend((u, v))
            cols.extend((v, u))
            weights.extend((weight, weight))

    # 地面点 p 的源节点 -> 可见卫星：仅计上行传播时延。
    # 可见卫星 -> 地面点 p 的目的节点：仅计下行传播时延。
    # 题目规定 0.5 ms/跳，因此只在每条星间链路上计一次处理时延。
    for p in range(point_count):
        source_node = TOTAL_SATS + p
        destination_node = TOTAL_SATS + point_count + p
        visible_satellites = np.flatnonzero(visible[p])

        for satellite in visible_satellites:
            satellite = int(satellite)

            rows.append(source_node)
            cols.append(satellite)
            weights.append(float(access_ms[p, satellite]))

            rows.append(satellite)
            cols.append(destination_node)
            weights.append(float(access_ms[p, satellite]))

    return csr_matrix(
        (weights, (rows, cols)),
        shape=(graph_size, graph_size),
    )


def reconstruct_path(predecessor_row, source_node, target_node):
    """根据 scipy 返回的前驱数组重建一条最短路径。"""
    path = []
    current = int(target_node)
    guard = 0

    while current != source_node:
        path.append(current)
        current = int(predecessor_row[current])
        guard += 1

        if current < 0 or guard > len(predecessor_row):
            return []

    path.append(source_node)
    path.reverse()
    return path


def satellite_label(satellite_id):
    """把卫星节点编号转换为 Plane-Satellite 标签。"""
    plane = satellite_id // N
    index = satellite_id % N
    return f"P{plane:02d}-S{index:02d}"


def analyse_worst_route(worst_route, points):
    """计算并保存最坏时延路径的距离、跳数与处理时延分解。"""
    if worst_route is None:
        return

    t = worst_route["time_s"]
    source_index = worst_route["source_index"]
    target_index = worst_route["target_index"]
    satellite_path = worst_route["satellite_path"]

    pos = satellite_positions(t).reshape(-1, 3)
    ground_pos = ground_positions_eci(points, t)

    first_satellite = satellite_path[0]
    last_satellite = satellite_path[-1]

    uplink_km = float(np.linalg.norm(
        ground_pos[source_index] - pos[first_satellite]
    ))
    downlink_km = float(np.linalg.norm(
        pos[last_satellite] - ground_pos[target_index]
    ))

    isl_km = 0.0
    for u, v in zip(satellite_path[:-1], satellite_path[1:]):
        isl_km += float(np.linalg.norm(pos[u] - pos[v]))

    total_distance_km = uplink_km + isl_km + downlink_km
    propagation_ms = total_distance_km / C_LIGHT * 1000.0
    processing_ms = max(len(satellite_path) - 1, 0) * PROCESSING_DELAY_MS
    reconstructed_total_ms = propagation_ms + processing_ms

    source_lat, source_lon = points[source_index]
    target_lat, target_lon = points[target_index]

    lines = [
        "Q3(2) Worst-case minimum-delay route",
        "=" * 72,
        f"Time: {t:.3f} s = {t / 60:.3f} min",
        f"Source ground point: ({source_lat:.3f} deg N, "
        f"{source_lon:.3f} deg E)",
        f"Target ground point: ({target_lat:.3f} deg N, "
        f"{target_lon:.3f} deg E)",
        f"Dijkstra latency: {worst_route['delay_ms']:.6f} ms",
        f"Reconstructed latency: {reconstructed_total_ms:.6f} ms",
        f"Uplink distance: {uplink_km:.3f} km",
        f"ISL path distance: {isl_km:.3f} km",
        f"Downlink distance: {downlink_km:.3f} km",
        f"Total propagation distance: {total_distance_km:.3f} km",
        f"Propagation delay: {propagation_ms:.6f} ms",
        f"Processing delay: {processing_ms:.6f} ms",
        f"Number of satellites on route: {len(satellite_path)}",
        f"Number of ISL hops: {max(0, len(satellite_path) - 1)}",
        "Satellite route:",
        " -> ".join(satellite_label(s) for s in satellite_path),
    ]

    path = os.path.join(OUTPUT_DIR, "Q3_2_worst_route.txt")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print(f"最坏路径明细已保存到：{path}")


def save_time_statistics(time_results):
    """保存各时刻的区域路由统计指标。"""
    data = np.column_stack((
        time_results["time_s"],
        time_results["time_s"] / 60.0,
        time_results["time_s"] / PERIOD,
        time_results["min_visible"],
        time_results["mean_visible"],
        time_results["max_visible"],
        time_results["mean_delay_ms"],
        time_results["p95_delay_ms"],
        time_results["max_delay_ms"],
        time_results["route_availability"],
        time_results["within_30ms_ratio"],
    ))

    header = (
        "time_s,time_min,orbit_fraction,"
        "min_visible_satellites,mean_visible_satellites,"
        "max_visible_satellites,mean_delay_ms,p95_delay_ms,"
        "max_delay_ms,route_availability,within_30ms_ratio"
    )

    path = os.path.join(OUTPUT_DIR, "Q3_2_time_statistics.csv")

    np.savetxt(
        path,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10f",
    )

    print(f"逐时刻统计数据已保存到：{path}")


def plot_results(time_results, all_delays):
    """绘制时延随时间变化曲线和总体经验分布函数。"""
    orbit_fraction = time_results["time_s"] / PERIOD

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        orbit_fraction,
        time_results["mean_delay_ms"],
        label="Mean latency",
    )
    plt.plot(
        orbit_fraction,
        time_results["p95_delay_ms"],
        label="95th percentile",
    )
    plt.plot(
        orbit_fraction,
        time_results["max_delay_ms"],
        label="Maximum latency",
    )
    plt.axhline(
        DELAY_LIMIT_MS,
        linestyle="--",
        label="30 ms requirement",
    )
    plt.xlabel("Orbital period fraction")
    plt.ylabel("End-to-end latency / ms")
    plt.title("Regional End-to-End Latency over One Orbital Period")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "Q3_2_latency_over_time.png"),
        dpi=300,
        bbox_inches="tight",
    )

    sorted_delays = np.sort(np.asarray(all_delays, dtype=float))
    cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)

    plt.figure(figsize=(8, 5.5))
    plt.plot(sorted_delays, cdf)
    plt.axvline(
        DELAY_LIMIT_MS,
        linestyle="--",
        label="30 ms requirement",
    )
    plt.xlabel("End-to-end latency / ms")
    plt.ylabel("Empirical cumulative probability")
    plt.title("CDF of Regional End-to-End Latency")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "Q3_2_latency_cdf.png"),
        dpi=300,
        bbox_inches="tight",
    )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


def run_routing_simulation():
    """执行一个轨道周期内的区域最小时延路由仿真。"""
    points = make_ground_grid()
    point_count = len(points)
    pair_indices = np.triu_indices(point_count, k=1)
    pair_count_per_time = len(pair_indices[0])

    # endpoint=False，避免周期末端与 t=0 重复。
    times = np.linspace(
        0.0,
        PERIOD,
        ROUTING_TIME_STEPS,
        endpoint=False,
    )

    positions_t0 = satellite_positions(0.0)
    intra_edges = build_intra_links(positions_t0)

    time_results = {
        "time_s": [],
        "min_visible": [],
        "mean_visible": [],
        "max_visible": [],
        "mean_delay_ms": [],
        "p95_delay_ms": [],
        "max_delay_ms": [],
        "route_availability": [],
        "within_30ms_ratio": [],
    }

    all_delays = []
    total_pair_samples = 0
    total_reachable_samples = 0
    total_within_limit_samples = 0
    worst_route = None
    global_max_delay = -np.inf

    print("=" * 72)
    print("Q3(2) 最小时延路由仿真")
    print("=" * 72)
    print(f"星座参数 M × N                = {M} × {N}")
    print(f"总卫星数量                    = {TOTAL_SATS}")
    print(f"轨道倾角                      = {I_DEG:.1f} deg")
    print(f"相位参数 F                    = {F}")
    print(f"轨道周期                      = {PERIOD / 60:.3f} min")
    print(f"目标区域网格步长              = {GROUND_GRID_STEP_DEG:.1f} deg")
    print(f"地面采样点数                  = {point_count}")
    print(f"每个时刻地面点对数量          = {pair_count_per_time}")
    print(f"时间采样点数                  = {ROUTING_TIME_STEPS}")
    print(f"单星地面覆盖半径              = {COVERAGE_RADIUS_KM:.1f} km")
    print(f"星上处理时延                  = {PROCESSING_DELAY_MS:.3f} ms/跳")
    print("=" * 72)

    for time_index, t in enumerate(times):
        pos = satellite_positions(float(t))
        sat_flat = pos.reshape(-1, 3)

        inter_edges = build_inter_links(pos)
        visible, access_ms = calculate_access_geometry(
            points,
            float(t),
            sat_flat,
        )

        visible_count = visible.sum(axis=1)
        graph = build_routing_graph(
            intra_edges,
            inter_edges,
            visible,
            access_ms,
        )

        source_nodes = TOTAL_SATS + np.arange(point_count)
        destination_nodes = (
            TOTAL_SATS + point_count + np.arange(point_count)
        )

        distance_matrix, predecessor = dijkstra(
            graph,
            directed=True,
            indices=source_nodes,
            return_predecessors=True,
        )

        ground_to_ground = distance_matrix[:, destination_nodes]
        pair_delays = ground_to_ground[pair_indices]
        finite_mask = np.isfinite(pair_delays)
        reachable_delays = pair_delays[finite_mask]

        reachable_count = int(np.count_nonzero(finite_mask))
        within_limit_count = int(np.count_nonzero(
            finite_mask & (pair_delays <= DELAY_LIMIT_MS)
        ))

        total_pair_samples += pair_count_per_time
        total_reachable_samples += reachable_count
        total_within_limit_samples += within_limit_count

        if reachable_count > 0:
            mean_delay = float(np.mean(reachable_delays))
            p95_delay = float(np.percentile(reachable_delays, 95))
            max_delay = float(np.max(reachable_delays))
            all_delays.extend(reachable_delays.tolist())
        else:
            mean_delay = np.nan
            p95_delay = np.nan
            max_delay = np.nan

        route_availability = reachable_count / pair_count_per_time
        within_limit_ratio = within_limit_count / pair_count_per_time

        time_results["time_s"].append(float(t))
        time_results["min_visible"].append(int(visible_count.min()))
        time_results["mean_visible"].append(float(visible_count.mean()))
        time_results["max_visible"].append(int(visible_count.max()))
        time_results["mean_delay_ms"].append(mean_delay)
        time_results["p95_delay_ms"].append(p95_delay)
        time_results["max_delay_ms"].append(max_delay)
        time_results["route_availability"].append(route_availability)
        time_results["within_30ms_ratio"].append(within_limit_ratio)

        # 保存全仿真中的最坏“仍可达”点对及其最短路径。
        if reachable_count > 0 and max_delay > global_max_delay:
            safe_delays = np.where(finite_mask, pair_delays, -np.inf)
            pair_position = int(np.argmax(safe_delays))
            source_index = int(pair_indices[0][pair_position])
            target_index = int(pair_indices[1][pair_position])

            source_node = int(source_nodes[source_index])
            target_node = int(destination_nodes[target_index])

            full_path = reconstruct_path(
                predecessor[source_index],
                source_node,
                target_node,
            )
            satellite_path = [
                node for node in full_path if node < TOTAL_SATS
            ]

            global_max_delay = max_delay
            worst_route = {
                "time_s": float(t),
                "source_index": source_index,
                "target_index": target_index,
                "delay_ms": max_delay,
                "satellite_path": satellite_path,
            }

        progress = 100.0 * (time_index + 1) / ROUTING_TIME_STEPS
        print(
            f"[{progress:6.1f}%] "
            f"t={t / 60:7.3f} min | "
            f"visible={visible_count.min():2d}~{visible_count.max():2d} | "
            f"mean={mean_delay:7.3f} ms | "
            f"P95={p95_delay:7.3f} ms | "
            f"max={max_delay:7.3f} ms | "
            f"<=30ms={within_limit_ratio * 100:6.2f}%"
        )

    for key in time_results:
        time_results[key] = np.asarray(time_results[key])

    all_delays_array = np.asarray(all_delays, dtype=float)

    overall_availability = (
        total_reachable_samples / total_pair_samples
    )
    overall_within_limit = (
        total_within_limit_samples / total_pair_samples
    )

    overall_mean = float(np.mean(all_delays_array))
    overall_p95 = float(np.percentile(all_delays_array, 95))
    overall_max = float(np.max(all_delays_array))

    strictly_satisfied = (
        np.isclose(overall_availability, 1.0)
        and overall_max <= DELAY_LIMIT_MS
    )

    print()
    print("=" * 72)
    print("Q3(2) 区域端到端时延统计结果")
    print("=" * 72)
    print(f"总地面点对—时刻样本数        = {total_pair_samples}")
    print(f"路由可达率                    = {overall_availability * 100:.4f}%")
    print(f"平均端到端时延                = {overall_mean:.4f} ms")
    print(f"95%分位端到端时延             = {overall_p95:.4f} ms")
    print(f"最大端到端时延                = {overall_max:.4f} ms")
    print(f"30 ms以内样本比例             = {overall_within_limit * 100:.4f}%")
    print("-" * 72)

    if strictly_satisfied:
        print("结论：在当前离散采样精度下，严格满足 30 ms 设计要求。")
    else:
        print("结论：在当前离散采样精度下，不能严格满足 30 ms 设计要求。")

        if overall_availability < 1.0:
            print("原因之一：部分地面点对在部分时刻不存在有效通信路径。")

        if overall_max > DELAY_LIMIT_MS:
            print(
                "原因之一：最坏场景的最小时延仍超过 30 ms，"
                "仅平均时延小于 30 ms 不足以判定满足要求。"
            )

    print("=" * 72)

    save_time_statistics(time_results)
    analyse_worst_route(worst_route, points)
    plot_results(time_results, all_delays_array)

    summary_path = os.path.join(OUTPUT_DIR, "Q3_2_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(
            f"M={M}, N={N}, i={I_DEG} deg, F={F}\n"
            f"Ground points={point_count}\n"
            f"Time samples={ROUTING_TIME_STEPS}\n"
            f"Route availability={overall_availability:.10f}\n"
            f"Mean latency={overall_mean:.10f} ms\n"
            f"95th percentile latency={overall_p95:.10f} ms\n"
            f"Maximum latency={overall_max:.10f} ms\n"
            f"Ratio within 30 ms={overall_within_limit:.10f}\n"
            f"Strictly satisfied={strictly_satisfied}\n"
        )

    print(f"汇总结果已保存到：{summary_path}")
    print(f"图像已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    run_routing_simulation()