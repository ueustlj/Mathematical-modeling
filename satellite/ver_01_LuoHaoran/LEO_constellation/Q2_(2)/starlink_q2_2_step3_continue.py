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


def make_ground_grid(lat_step=2.0, lon_step=2.0):
    lats = np.arange(LAT_MIN, LAT_MAX + 1e-9, lat_step)
    lons = np.arange(LON_MIN, LON_MAX + 1e-9, lon_step)

    lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")

    lat_rad = deg2rad(lat_mesh.ravel())
    lon_rad = deg2rad(lon_mesh.ravel())

    return lat_rad, lon_rad, lat_mesh, lon_mesh


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


def check_feasible(
    M,
    N,
    inc_deg,
    F,
    lat_step,
    lon_step,
    duration_hours,
    dt_seconds,
    return_detail=False
):
    ground_lats, ground_lons, lat_mesh, lon_mesh = make_ground_grid(lat_step, lon_step)
    times = np.arange(0, duration_hours * 3600 + 1e-9, dt_seconds)

    total_checks = 0
    covered_checks = 0

    global_min_count = 10**9
    worst_time = None
    worst_point_index = None
    worst_lat = None
    worst_lon = None

    for t in times:
        counts = coverage_counts(M, N, inc_deg, F, t, ground_lats, ground_lons)

        min_count = counts.min()

        if min_count < global_min_count:
            global_min_count = min_count
            worst_time = t
            worst_point_index = int(np.argmin(counts))
            worst_lat = np.rad2deg(ground_lats[worst_point_index])
            worst_lon = np.rad2deg(ground_lons[worst_point_index])

        total_checks += counts.size
        covered_checks += np.sum(counts >= 1)

        # 搜索时加速：只要发现 0 覆盖，直接判失败
        if not return_detail and min_count == 0:
            return False, global_min_count, covered_checks / total_checks, worst_time, worst_lat, worst_lon

    coverage_rate = covered_checks / total_checks
    feasible = global_min_count >= 1

    return feasible, global_min_count, coverage_rate, worst_time, worst_lat, worst_lon


# =========================
# 3. 从 1243 继续搜索
# =========================

def continue_search():
    print("=" * 80)
    print("Continue Search Start")
    print("=" * 80)

    # 第一层：粗筛选参数
    coarse_lat_step = 3.0
    coarse_lon_step = 3.0
    coarse_duration_hours = 6.0
    coarse_dt = 600.0

    # 第二层：中等验证参数
    fine_lat_step = 2.0
    fine_lon_step = 2.0
    fine_duration_hours = 6.0
    fine_dt = 300.0

    start_total = 1243
    max_total = 1800

    M_values = range(8, 71)
    N_values = range(8, 91)

    # 因为目标区域最高到 53°N，倾角太低不利于覆盖北边界
    inc_values = range(48, 61, 1)

    start_time = time.time()
    checked = 0
    coarse_passed = 0
    fine_passed = []

    print(f"Search total satellites from {start_total} to {max_total}")
    print(f"Coarse check: {coarse_lat_step} deg, {coarse_dt} s")
    print(f"Fine check:   {fine_lat_step} deg, {fine_dt} s")
    print("=" * 80)

    for total_sat in range(start_total, max_total + 1):

        found_at_this_total = []

        for M in M_values:
            if total_sat % M != 0:
                continue

            N = total_sat // M

            if N not in N_values:
                continue

            for inc_deg in inc_values:
                for F in range(M):
                    checked += 1

                    coarse_ok, coarse_min, coarse_rate, coarse_wt, coarse_wlat, coarse_wlon = check_feasible(
                        M, N, inc_deg, F,
                        coarse_lat_step,
                        coarse_lon_step,
                        coarse_duration_hours,
                        coarse_dt,
                        return_detail=False
                    )

                    if not coarse_ok:
                        continue

                    coarse_passed += 1

                    print("\nCoarse feasible candidate:")
                    print(f"M={M}, N={N}, i={inc_deg}, F={F}, total={total_sat}")

                    fine_ok, fine_min, fine_rate, fine_wt, fine_wlat, fine_wlon = check_feasible(
                        M, N, inc_deg, F,
                        fine_lat_step,
                        fine_lon_step,
                        fine_duration_hours,
                        fine_dt,
                        return_detail=True
                    )

                    print("Fine validation:")
                    print(f"  Minimum coverage count = {fine_min}")
                    print(f"  Single coverage rate = {fine_rate * 100:.6f}%")
                    print(f"  Worst time = {fine_wt / 3600:.3f} h")
                    print(f"  Worst point = ({fine_wlat:.2f} N, {fine_wlon:.2f} E)")
                    print(f"  Feasible = {fine_ok}")

                    if fine_ok:
                        result = {
                            "M": M,
                            "N": N,
                            "inc": inc_deg,
                            "F": F,
                            "total": total_sat,
                            "min_count": fine_min,
                            "coverage_rate": fine_rate,
                            "worst_time_h": fine_wt / 3600,
                            "worst_lat": fine_wlat,
                            "worst_lon": fine_wlon
                        }

                        fine_passed.append(result)
                        found_at_this_total.append(result)

        if len(found_at_this_total) > 0:
            print("\n" + "=" * 80)
            print("Fine feasible solution found!")
            print("=" * 80)
            print(f"Minimum total satellites = {total_sat}")
            print("Solutions:")
            for r in found_at_this_total:
                print(r)
            print("=" * 80)
            break

        if total_sat % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"Progress: total_sat={total_sat}, checked={checked}, "
                f"coarse_passed={coarse_passed}, elapsed={elapsed:.1f} s"
            )

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("Continue Search Finished")
    print("=" * 80)
    print(f"Checked cases = {checked}")
    print(f"Coarse passed = {coarse_passed}")
    print(f"Fine passed = {len(fine_passed)}")
    print(f"Elapsed time = {elapsed:.1f} s")
    print("=" * 80)

    if len(fine_passed) > 0:
        print("Best fine feasible solution:")
        print(fine_passed[0])
    else:
        print("No fine feasible solution found in this range.")


if __name__ == "__main__":
    continue_search()