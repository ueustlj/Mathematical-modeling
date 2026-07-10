import numpy as np
import time

# =========================
# 1. 基本常数
# =========================

R_E = 6371.0
H = 550.0
A = R_E + H
MU = 398600.4418
OMEGA_E = 7.2921159e-5

COVER_RADIUS = 506.0
COVER_ANGLE = COVER_RADIUS / R_E
COS_COVER = np.cos(COVER_ANGLE)

LAT_MIN, LAT_MAX = 4.0, 53.0
LON_MIN, LON_MAX = 73.0, 135.0


# =========================
# 2. 工具函数
# =========================

def deg2rad(x):
    return np.deg2rad(x)


def make_ground_grid(lat_step=3.0, lon_step=3.0):
    lats = np.arange(LAT_MIN, LAT_MAX + 1e-9, lat_step)
    lons = np.arange(LON_MIN, LON_MAX + 1e-9, lon_step)

    lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")

    lat_rad = deg2rad(lat_mesh.ravel())
    lon_rad = deg2rad(lon_mesh.ravel())

    return lat_rad, lon_rad


def satellite_subpoints(M, N, inc_deg, F, t):
    inc = deg2rad(inc_deg)
    mean_motion = np.sqrt(MU / A**3)

    total_sat = M * N
    sat_lats = np.empty(total_sat)
    sat_lons = np.empty(total_sat)

    idx = 0

    for m in range(M):
        raan = 2 * np.pi * m / M

        for n in range(N):
            u0 = 2 * np.pi * n / N + 2 * np.pi * F * m / (M * N)
            u = u0 + mean_motion * t

            lat = np.arcsin(np.sin(inc) * np.sin(u))

            lon_inertial = raan + np.arctan2(
                np.cos(inc) * np.sin(u),
                np.cos(u)
            )

            lon = lon_inertial - OMEGA_E * t
            lon = (lon + np.pi) % (2 * np.pi) - np.pi

            sat_lats[idx] = lat
            sat_lons[idx] = lon
            idx += 1

    return sat_lats, sat_lons


def coverage_counts(M, N, inc_deg, F, t, ground_lats, ground_lons):
    sat_lats, sat_lons = satellite_subpoints(M, N, inc_deg, F, t)

    sin_g = np.sin(ground_lats)[:, None]
    cos_g = np.cos(ground_lats)[:, None]

    sin_s = np.sin(sat_lats)[None, :]
    cos_s = np.cos(sat_lats)[None, :]

    delta_lon = ground_lons[:, None] - sat_lons[None, :]

    cos_theta = sin_g * sin_s + cos_g * cos_s * np.cos(delta_lon)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    covered = cos_theta >= COS_COVER
    counts = covered.sum(axis=1)

    return counts


def is_feasible_fast(
    M,
    N,
    inc_deg,
    F,
    ground_lats,
    ground_lons,
    times
):
    """
    快速判断是否可行。
    一旦发现某个时刻有 0 覆盖点，立刻返回 False。
    """
    min_count_all = 10**9
    worst_time = None

    for t in times:
        counts = coverage_counts(M, N, inc_deg, F, t, ground_lats, ground_lons)
        min_count = counts.min()

        if min_count < min_count_all:
            min_count_all = min_count
            worst_time = t

        if min_count == 0:
            return False, min_count_all, worst_time

    return True, min_count_all, worst_time


def evaluate_detail(
    M,
    N,
    inc_deg,
    F,
    lat_step=2.0,
    lon_step=2.0,
    duration_hours=6.0,
    dt_seconds=300.0
):
    """
    对候选方案做详细统计。
    """
    ground_lats, ground_lons = make_ground_grid(lat_step, lon_step)
    times = np.arange(0, duration_hours * 3600 + 1e-9, dt_seconds)

    total_checks = 0
    covered_checks = 0
    global_min_count = 10**9
    worst_time = None
    worst_point_index = None

    for t in times:
        counts = coverage_counts(M, N, inc_deg, F, t, ground_lats, ground_lons)

        min_count = counts.min()
        if min_count < global_min_count:
            global_min_count = min_count
            worst_time = t
            worst_point_index = int(np.argmin(counts))

        total_checks += counts.size
        covered_checks += np.sum(counts >= 1)

    coverage_rate = covered_checks / total_checks
    feasible = global_min_count >= 1

    print("=" * 70)
    print("Detailed Evaluation")
    print("=" * 70)
    print(f"M = {M}")
    print(f"N = {N}")
    print(f"inclination = {inc_deg} deg")
    print(f"F = {F}")
    print(f"Total satellites = {M * N}")
    print("-" * 70)
    print(f"Grid step = {lat_step} deg × {lon_step} deg")
    print(f"Duration = {duration_hours} h")
    print(f"Time step = {dt_seconds} s")
    print(f"Ground points = {len(ground_lats)}")
    print(f"Time samples = {len(times)}")
    print("-" * 70)
    print(f"Minimum coverage count = {global_min_count}")
    print(f"Single coverage rate = {coverage_rate * 100:.6f}%")
    print(f"Worst time = {worst_time / 3600:.3f} h")
    print(f"Worst point index = {worst_point_index}")
    print(f"Feasible = {feasible}")
    print("=" * 70)

    return feasible, coverage_rate, global_min_count


# =========================
# 3. 粗搜索
# =========================

def coarse_search():
    """
    第一轮粗搜索：
    网格较粗，时间步较大，用来快速找候选解。
    """

    # 粗搜索参数
    lat_step = 3.0
    lon_step = 3.0
    duration_hours = 6.0
    dt_seconds = 600.0

    ground_lats, ground_lons = make_ground_grid(lat_step, lon_step)
    times = np.arange(0, duration_hours * 3600 + 1e-9, dt_seconds)

    print("=" * 70)
    print("Coarse Search Start")
    print("=" * 70)
    print(f"Grid step = {lat_step} deg × {lon_step} deg")
    print(f"Duration = {duration_hours} h")
    print(f"Time step = {dt_seconds} s")
    print(f"Ground points = {len(ground_lats)}")
    print(f"Time samples = {len(times)}")
    print("=" * 70)

    start_time = time.time()

    candidates = []

    # 搜索范围
    M_values = range(8, 61)       # 轨道面数
    N_values = range(8, 81)       # 每面卫星数
    inc_values = range(48, 61, 2) # 倾角。目标最高到 53N，所以先搜 48~60
    max_total_satellites = 2500

    checked = 0

    # 按总卫星数从小到大搜索
    for total_sat in range(200, max_total_satellites + 1):

        found_this_total = []

        for M in M_values:
            if total_sat % M != 0:
                continue

            N = total_sat // M

            if N not in N_values:
                continue

            # F 是相位因子，范围 0 到 M-1
            for inc_deg in inc_values:
                for F in range(M):
                    checked += 1

                    feasible, min_count, worst_time = is_feasible_fast(
                        M, N, inc_deg, F,
                        ground_lats, ground_lons, times
                    )

                    if feasible:
                        result = {
                            "M": M,
                            "N": N,
                            "inc": inc_deg,
                            "F": F,
                            "total": total_sat,
                            "min_count": min_count,
                            "worst_time": worst_time
                        }

                        candidates.append(result)
                        found_this_total.append(result)

                        print("\nFound feasible candidate:")
                        print(result)

        # 如果当前总卫星数已经找到可行解，粗搜索可以停止
        if len(found_this_total) > 0:
            print("\n" + "=" * 70)
            print("Minimum total satellites found in coarse search.")
            print("=" * 70)
            print(f"Total satellites = {total_sat}")
            print(f"Number of candidates = {len(found_this_total)}")
            print("=" * 70)
            break

        if total_sat % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Checked up to total_sat = {total_sat}, checked = {checked}, elapsed = {elapsed:.1f} s")

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("Coarse Search Finished")
    print("=" * 70)
    print(f"Checked cases = {checked}")
    print(f"Elapsed time = {elapsed:.1f} s")
    print(f"Candidates found = {len(candidates)}")
    print("=" * 70)

    if len(candidates) > 0:
        print("\nBest candidates:")
        for c in candidates[:10]:
            print(c)

    return candidates


# =========================
# 4. 主程序
# =========================

if __name__ == "__main__":
    candidates = coarse_search()

    if len(candidates) > 0:
        best = candidates[0]

        print("\nNow evaluating the first candidate in more detail...\n")

        evaluate_detail(
            best["M"],
            best["N"],
            best["inc"],
            best["F"],
            lat_step=2.0,
            lon_step=2.0,
            duration_hours=6.0,
            dt_seconds=300.0
        )