"""
Roughness-based crevasse candidate screening for ONE ICESat-2 ATL03 track.

Workflow
--------
1. Read one preprocessed ATL03 TXT file.
2. Select one possible signal photon (maximum weight) in each 3 m segment.
3. Remove possible signal photons whose elevation differs by > 1 m from the
   linear interpolation between their two neighboring possible signal photons.
4. Divide the track into 150 m windows and select the maximum-elevation
   possible signal photon in each window as a trend-surface control photon.
5. For each pair of neighboring trend-surface control photons, calculate the
   perpendicular distances from the possible signal photons P between them
   to the line connecting the two trend-surface control photons.
6. Calculate roughness as the population standard deviation of the perpendicular
   distances:

       R_q = sqrt( sum((d_i - mean(d))**2) / n )
7. Mark intervals with roughness >= 1.0 m as candidate crevasse regions.
8. Expand each candidate interval by 150 m on both sides and merge overlaps.
9. Save candidate photons and diagnostic files.

Input column layout
-------------------
Expected input column layout:
    column 0: along-track distance (m)
    column 1: elevation (m)
    column 2: photon weight
    column 3: UTM easting
    column 4: UTM northing
    column 5: longitude
    column 6: latitude

The input file is assumed to be preprocessed in advance, as in the original
workflow. In particular, any upstream ATL03 confidence filtering should
already have been applied.
"""

from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

# Set the path to ONE preprocessed ATL03 TXT file.
INPUT_FILE = Path(r"20190112_0224_gt1l.txt")

# Output directory.
OUTPUT_DIR = Path("roughness_output")

# Input columns (zero-based; consistent with the original code).
ALONG_COL = 0
ELEVATION_COL = 1
WEIGHT_COL = 2

# Output columns retained for the next processing step.
# This follows the original output order:
# along-track distance, elevation, easting, northing, longitude, latitude.
OUTPUT_COLUMNS = [0, 1, 3, 4, 5, 6]

# Method parameters.
SMALL_WINDOW_M = 3.0
SMOOTHNESS_THRESHOLD_M = 1.0
TREND_WINDOW_M = 150.0
ROUGHNESS_THRESHOLD_M = 1.0
BUFFER_EACH_SIDE_M = 150.0

# Plot settings.
SAVE_PLOT = True
PLOT_DPI = 300


# =============================================================================
# 2. DATA AND WINDOW UTILITIES
# =============================================================================

def load_data(file_path: Path) -> np.ndarray:
    """Load and sort one comma-delimited preprocessed ATL03 TXT file."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    data = np.loadtxt(
        file_path,
        delimiter=",",
        dtype=np.float64,
        ndmin=2
    )

    required_col = max(ALONG_COL, ELEVATION_COL, WEIGHT_COL)
    if data.shape[1] <= required_col:
        raise ValueError(
            f"The input file has {data.shape[1]} columns; "
            f"at least {required_col + 1} columns are required."
        )

    # Remove rows with invalid values in the columns used by this module.
    valid = np.all(
        np.isfinite(data[:, [ALONG_COL, ELEVATION_COL, WEIGHT_COL]]),
        axis=1
    )
    data = data[valid]

    if data.size == 0:
        raise ValueError("No valid photons were found in the input file.")

    return data[np.argsort(data[:, ALONG_COL])]


def find_window_starts(along_track: np.ndarray, window_m: float) -> np.ndarray:
    """
    Reproduce the node-selection logic of the original code:
    the first photon is the first node, and each following node is the first
    photon at least window_m beyond the previous node.
    """
    if along_track.size == 0:
        return np.empty(0, dtype=np.int64)

    node_indices = [0]
    last_distance = along_track[0]

    for i in range(1, len(along_track)):
        if along_track[i] - last_distance >= window_m:
            node_indices.append(i)
            last_distance = along_track[i]

    return np.asarray(node_indices, dtype=np.int64)


def merge_intervals(intervals):
    """Merge overlapping or touching along-track intervals."""
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda item: item[0])
    merged = [[float(intervals[0][0]), float(intervals[0][1])]]

    for start, end in intervals[1:]:
        start = float(start)
        end = float(end)

        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(item[0], item[1]) for item in merged]


# =============================================================================
# 3. POSSIBLE SIGNAL PHOTON SELECTION: 3 m WINDOW
# =============================================================================

def select_possible_signal_photons(data: np.ndarray):
    """
    Select the maximum-weight photon in each 3 m segment.

    Returns
    -------
    possible : ndarray, shape (n, 3)
        Columns are along-track distance, elevation, and weight.
    """
    along = data[:, ALONG_COL]
    elevation = data[:, ELEVATION_COL]
    weight = data[:, WEIGHT_COL]

    node_indices = find_window_starts(along, SMALL_WINDOW_M)
    selected_rows = []

    for node_idx in node_indices:
        start = along[node_idx]
        end = start + SMALL_WINDOW_M

        left = np.searchsorted(along, start, side="left")
        right = np.searchsorted(along, end, side="left")

        if right <= left:
            continue

        local_idx = left + int(np.argmax(weight[left:right]))
        selected_rows.append([
            along[local_idx],
            elevation[local_idx],
            weight[local_idx]
        ])

    if not selected_rows:
        raise ValueError("No possible signal photons were selected.")

    possible = np.asarray(selected_rows, dtype=np.float64)
    possible = possible[np.argsort(possible[:, 0])]

    return possible


# =============================================================================
# 4. SMOOTHNESS-BASED OUTLIER REMOVAL
# =============================================================================

def remove_smoothness_outliers(
    possible: np.ndarray,
    threshold_m: float = SMOOTHNESS_THRESHOLD_M
):
    """
    Remove an interior possible signal photon if its elevation differs by
    more than threshold_m from the linear interpolation between its two
    neighboring possible signal photons.

    The first and last photons are retained because a two-sided interpolation
    test cannot be formed for them.

    Returns
    -------
    filtered : ndarray
        Possible signal photons retained after filtering.
    removed : ndarray
        Possible signal photons identified as outliers.
    keep_mask : ndarray[bool]
        Retention flag for each original possible signal photon.
    deviations : ndarray
        Absolute elevation deviations; NaN for the two endpoints.
    """
    n = len(possible)
    keep_mask = np.ones(n, dtype=bool)
    deviations = np.full(n, np.nan, dtype=np.float64)

    if n < 3:
        return possible.copy(), possible[:0].copy(), keep_mask, deviations

    x = possible[:, 0]
    z = possible[:, 1]

    # Evaluate all interior photons against the original neighboring set.
    # Outliers are removed simultaneously after all deviations are computed.
    for i in range(1, n - 1):
        x_left, x_mid, x_right = x[i - 1], x[i], x[i + 1]

        if x_right <= x_left:
            continue

        ratio = (x_mid - x_left) / (x_right - x_left)
        expected_elevation = (
            z[i - 1] + ratio * (z[i + 1] - z[i - 1])
        )

        deviations[i] = abs(z[i] - expected_elevation)

        if deviations[i] > threshold_m:
            keep_mask[i] = False

    filtered = possible[keep_mask]
    removed = possible[~keep_mask]

    return filtered, removed, keep_mask, deviations


# =============================================================================
# 5. TREND-SURFACE CONTROL PHOTONS: 150 m WINDOW
# =============================================================================

def select_trend_surface_photons(
    data: np.ndarray,
    possible: np.ndarray
):
    """
    Divide the input track into 150 m windows. Within each window, select the
    maximum-elevation photon from the filtered possible signal photon set P.

    Returns
    -------
    trend : ndarray, shape (m, 2)
        Along-track distance and elevation of trend-surface control photons E.
    window_starts : ndarray
        Along-track start of each 150 m window that yielded a control photon.
    """
    input_along = data[:, ALONG_COL]
    possible_along = possible[:, 0]
    possible_elevation = possible[:, 1]

    node_indices = find_window_starts(input_along, TREND_WINDOW_M)

    trend_rows = []
    valid_window_starts = []

    for node_idx in node_indices:
        start = input_along[node_idx]
        end = start + TREND_WINDOW_M

        left = np.searchsorted(possible_along, start, side="left")
        right = np.searchsorted(possible_along, end, side="left")

        if right <= left:
            continue

        local_idx = left + int(np.argmax(possible_elevation[left:right]))

        trend_rows.append([
            possible_along[local_idx],
            possible_elevation[local_idx]
        ])
        valid_window_starts.append(start)

    if len(trend_rows) < 2:
        raise ValueError(
            "Fewer than two trend-surface control photons were obtained. "
            "At least two neighboring control photons are required."
        )

    trend = np.asarray(trend_rows, dtype=np.float64)
    window_starts = np.asarray(valid_window_starts, dtype=np.float64)

    order = np.argsort(trend[:, 0])
    return trend[order], window_starts[order]


# =============================================================================
# 6. ROUGHNESS BETWEEN NEIGHBORING TREND-SURFACE PHOTONS
# =============================================================================

def perpendicular_distances_to_line(
    x: np.ndarray,
    z: np.ndarray,
    x1: float,
    z1: float,
    x2: float,
    z2: float
) -> np.ndarray:
    """Calculate absolute perpendicular distances from points to a line."""
    dx = x2 - x1
    dz = z2 - z1
    denominator = np.hypot(dx, dz)

    if denominator == 0:
        return np.hypot(x - x1, z - z1)

    return np.abs(
        dz * x - dx * z + x2 * z1 - z2 * x1
    ) / denominator


def calculate_roughness(
    possible: np.ndarray,
    trend: np.ndarray,
    trend_window_starts: np.ndarray
):
    """
    Calculate roughness using ONLY the filtered possible signal photons P
    between each pair of neighboring trend-surface control photons E.

    Formula:
        Rq = sqrt(sum((d_i - d_bar)^2) / n)

    This is the population standard deviation of the perpendicular distances.
    """
    p_x = possible[:, 0]
    p_z = possible[:, 1]

    records = []

    for i in range(len(trend) - 1):
        x1, z1 = trend[i]
        x2, z2 = trend[i + 1]

        if x2 <= x1:
            continue

        # Possible signal photons P located between two neighboring E photons.
        left = np.searchsorted(p_x, x1, side="left")
        right = np.searchsorted(p_x, x2, side="right")

        interval_x = p_x[left:right]
        interval_z = p_z[left:right]

        if interval_x.size < 2:
            continue

        distances = perpendicular_distances_to_line(
            interval_x, interval_z, x1, z1, x2, z2
        )

        mean_distance = float(np.mean(distances))
        roughness = float(np.sqrt(
            np.sum((distances - mean_distance) ** 2) / distances.size
        ))

        records.append({
            "interval_start_m": float(x1),
            "interval_end_m": float(x2),
            "n_possible_signal_photons": int(distances.size),
            "mean_distance_m": mean_distance,
            "roughness_m": roughness,
            "is_candidate": bool(roughness >= ROUGHNESS_THRESHOLD_M),
        })

    if not records:
        raise ValueError(
            "No valid roughness intervals were calculated. "
            "Check the possible signal photons and trend control photons."
        )

    return records


# =============================================================================
# 7. CANDIDATE REGION EXPANSION AND EXTRACTION
# =============================================================================

def extract_candidate_regions(
    data: np.ndarray,
    roughness_records: list
):
    """
    For each interval with roughness >= threshold, expand the interval by
    150 m on both sides. Merge overlapping buffered intervals and select all
    original input photons within the merged candidate regions.
    """
    track_min = float(data[:, ALONG_COL].min())
    track_max = float(data[:, ALONG_COL].max())

    raw_candidate_intervals = [
        (
            record["interval_start_m"],
            record["interval_end_m"]
        )
        for record in roughness_records
        if record["is_candidate"]
    ]

    buffered_intervals = []
    for start, end in raw_candidate_intervals:
        buffered_intervals.append((
            max(track_min, start - BUFFER_EACH_SIDE_M),
            min(track_max, end + BUFFER_EACH_SIDE_M)
        ))

    merged_regions = merge_intervals(buffered_intervals)

    along = data[:, ALONG_COL]
    candidate_mask = np.zeros(len(data), dtype=bool)

    for start, end in merged_regions:
        left = np.searchsorted(along, start, side="left")
        right = np.searchsorted(along, end, side="right")
        candidate_mask[left:right] = True

    return raw_candidate_intervals, merged_regions, candidate_mask


# =============================================================================
# 8. OUTPUT AND VISUALIZATION
# =============================================================================

def save_diagnostic_plot(
    data: np.ndarray,
    possible_before: np.ndarray,
    possible_after: np.ndarray,
    removed: np.ndarray,
    trend: np.ndarray,
    roughness_records: list,
    candidate_mask: np.ndarray,
    output_path: Path
):
    """Save an overview plot of the roughness-based screening results."""
    x = data[:, ALONG_COL]
    z = data[:, ELEVATION_COL]

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.scatter(
        x, z, s=1, color="0.70", alpha=0.4,
        label="Input photons"
    )

    if candidate_mask.any():
        ax.scatter(
            x[candidate_mask], z[candidate_mask],
            s=2, color="tab:red", alpha=0.6,
            label="Buffered candidate regions"
        )

    ax.scatter(
        possible_before[:, 0], possible_before[:, 1],
        s=8, marker="x", color="tab:orange",
        label="Possible signal photons before filtering"
    )

    if removed.size:
        ax.scatter(
            removed[:, 0], removed[:, 1],
            s=24, marker="x", color="red",
            label="Removed outliers (> 1 m)"
        )

    ax.scatter(
        possible_after[:, 0], possible_after[:, 1],
        s=9, facecolors="none", edgecolors="tab:blue",
        label="Possible signal photons after filtering"
    )

    ax.plot(
        trend[:, 0], trend[:, 1],
        color="black", linewidth=1.2,
        marker="o", markersize=3,
        label="Trend-surface control photons"
    )

    ax.set_xlabel("Along-track distance (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Roughness-based crevasse candidate screening")
    ax.legend(loc="best", frameon=False, fontsize=8)

    # Plot the roughness values on a secondary y-axis.
    ax2 = ax.twinx()
    rough_x = [
        (r["interval_start_m"] + r["interval_end_m"]) / 2
        for r in roughness_records
    ]
    rough_y = [r["roughness_m"] for r in roughness_records]

    ax2.plot(
        rough_x, rough_y,
        color="tab:green", linestyle="--",
        linewidth=1.1, marker=".", markersize=4,
        label="Roughness"
    )
    ax2.axhline(
        ROUGHNESS_THRESHOLD_M,
        color="tab:purple", linestyle=":",
        linewidth=1.2, label="Threshold = 1 m"
    )
    ax2.set_ylabel("Roughness (m)")
    ax2.legend(loc="upper right", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_roughness_csv(records: list, output_path: Path):
    header = (
        "interval_start_m,interval_end_m,n_possible_signal_photons,"
        "mean_distance_m,roughness_m,is_candidate"
    )

    values = np.asarray([
        [
            r["interval_start_m"],
            r["interval_end_m"],
            r["n_possible_signal_photons"],
            r["mean_distance_m"],
            r["roughness_m"],
            int(r["is_candidate"])
        ]
        for r in records
    ], dtype=np.float64)

    np.savetxt(
        output_path,
        values,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.8f"
    )


def main():
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read one preprocessed ATL03 track.
    data = load_data(INPUT_FILE)
    print(f"Input file: {INPUT_FILE}")
    print(f"Input photon count: {len(data):,}")

    # Step 1: select possible signal photons every 3 m.
    possible_before = select_possible_signal_photons(data)
    print(
        "Possible signal photons before smoothness filtering: "
        f"{len(possible_before):,}"
    )

    # Step 2: remove possible signal photons with >1 m local elevation deviation.
    possible_after, removed, keep_mask, deviations = (
        remove_smoothness_outliers(
            possible_before,
            SMOOTHNESS_THRESHOLD_M
        )
    )
    print(f"Removed anomalous photons: {len(removed):,}")
    print(
        "Possible signal photons after smoothness filtering: "
        f"{len(possible_after):,}"
    )

    # Step 3: construct trend-surface control set E from 150 m windows.
    trend, trend_window_starts = select_trend_surface_photons(
        data,
        possible_after
    )
    print(f"Trend-surface control photons: {len(trend):,}")

    # Step 4: calculate roughness from P photons between neighboring E photons.
    roughness_records = calculate_roughness(
        possible_after,
        trend,
        trend_window_starts
    )

    # Step 5: identify and expand candidate regions.
    raw_intervals, merged_regions, candidate_mask = extract_candidate_regions(
        data,
        roughness_records
    )
    candidate_data = data[candidate_mask]

    # Step 6: save candidate ATL03 photons in the original downstream format.
    if max(OUTPUT_COLUMNS) >= data.shape[1]:
        raise ValueError(
            "The input file does not contain all requested output columns. "
            "Check OUTPUT_COLUMNS and the input TXT column layout."
        )

    candidate_output = candidate_data[:, OUTPUT_COLUMNS]
    stem = INPUT_FILE.stem

    candidate_path = OUTPUT_DIR / f"{stem}_roughness_candidates.txt"
    np.savetxt(
        candidate_path,
        candidate_output,
        delimiter=",",
        fmt="%.10f"
    )

    # Save possible signal photons and outlier flags.
    possible_path = OUTPUT_DIR / f"{stem}_possible_signal_photons.csv"
    possible_table = np.column_stack((
        possible_before,
        keep_mask.astype(np.int32),
        deviations
    ))
    np.savetxt(
        possible_path,
        possible_table,
        delimiter=",",
        header=(
            "along_track_m,elevation_m,weight,retained,"
            "deviation_from_neighbor_interpolation_m"
        ),
        comments="",
        fmt="%.8f"
    )

    # Save removed outliers.
    removed_path = OUTPUT_DIR / f"{stem}_removed_outliers.csv"
    np.savetxt(
        removed_path,
        removed,
        delimiter=",",
        header="along_track_m,elevation_m,weight",
        comments="",
        fmt="%.8f"
    )

    # Save trend-surface control photons.
    trend_path = OUTPUT_DIR / f"{stem}_trend_surface_photons.csv"
    np.savetxt(
        trend_path,
        trend,
        delimiter=",",
        header="along_track_m,elevation_m",
        comments="",
        fmt="%.8f"
    )

    # Save roughness metrics.
    roughness_path = OUTPUT_DIR / f"{stem}_roughness_metrics.csv"
    save_roughness_csv(roughness_records, roughness_path)

    # Save candidate-region extents.
    region_path = OUTPUT_DIR / f"{stem}_candidate_regions.csv"
    with open(region_path, "w", encoding="utf-8") as f:
        f.write(
            "candidate_interval_start_m,candidate_interval_end_m,"
            "buffered_region_start_m,buffered_region_end_m\n"
        )

        # Each raw candidate interval is expanded independently here.
        track_min = float(data[:, ALONG_COL].min())
        track_max = float(data[:, ALONG_COL].max())

        for start, end in raw_intervals:
            buffered_start = max(
                track_min, start - BUFFER_EACH_SIDE_M
            )
            buffered_end = min(
                track_max, end + BUFFER_EACH_SIDE_M
            )
            f.write(
                f"{start:.3f},{end:.3f},"
                f"{buffered_start:.3f},{buffered_end:.3f}\n"
            )

    # Save diagnostic plot.
    if SAVE_PLOT:
        plot_path = OUTPUT_DIR / f"{stem}_roughness_screening.png"
        save_diagnostic_plot(
            data,
            possible_before,
            possible_after,
            removed,
            trend,
            roughness_records,
            candidate_mask,
            plot_path
        )
        print(f"Diagnostic plot: {plot_path}")

    elapsed = time.perf_counter() - start_time

    n_above_threshold = sum(
        record["is_candidate"] for record in roughness_records
    )

    print("\n========== Screening completed ==========")
    print(f"Roughness intervals evaluated: {len(roughness_records):,}")
    print(f"Intervals above threshold: {n_above_threshold:,}")
    print(f"Merged candidate regions: {len(merged_regions):,}")
    print(f"Candidate input photons: {len(candidate_data):,}")
    print(f"Candidate output: {candidate_path}")
    print(f"Possible photons: {possible_path}")
    print(f"Removed outliers: {removed_path}")
    print(f"Trend control photons: {trend_path}")
    print(f"Roughness metrics: {roughness_path}")
    print(f"Candidate regions: {region_path}")
    print(f"Elapsed time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
