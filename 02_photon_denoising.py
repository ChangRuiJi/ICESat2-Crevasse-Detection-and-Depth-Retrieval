"""
Direction-adaptive photon denoising for ONE ICESat-2 ATL03 roughness candidate file.

Input
-----
This script is designed to read the candidate-photon TXT file produced by
Roughness.py.

Expected input columns (comma-delimited, no header):
    0: along-track distance (m)
    1: elevation (m)
    2: UTM easting (m)
    3: UTM northing (m)
    4: longitude (degrees)
    5: latitude (degrees)

Workflow
--------
1. Read and sort the roughness-screened candidate photons.
2. Calculate the mean distance to the 20 nearest neighbors in the
   along-track/elevation space.
3. Estimate local glacier-surface direction using RANSAC in 10 m steps,
   with a +/-10 m fitting window.
4. Calculate direction-adaptive similarity-weighted density in an elliptical
   neighborhood (semi-axes: 10 m and 0.7 m).
5. Apply the adaptive density threshold in overlapping 3 m windows:
       threshold = max(mean(local density) - 0.1, 0)
6. Merge duplicate retained photons.
7. Apply the optional local spatial-support filter to remove isolated
   retained photons.
8. Save denoised photons in the same six-column format as the input, so the
   output can be passed to the subsequent fitting/crevasse-extraction module.

Dependencies
------------
numpy
matplotlib
scikit-learn

Run
---
1. Place the roughness candidate TXT file in the roughness_output folder.
2. Set INPUT_FILE below to that file.
3. Run this script.
"""

from pathlib import Path
import math
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

# Keep these paths consistent with the roughness-screening script.
# Run both scripts from the repository root directory.
INPUT_FILE = (
    Path("roughness_output")
    / "20190112_0224_gt1l_roughness_candidates.txt"
)

# Output directory.
OUTPUT_DIR = Path("denoising_output")

# Input column definitions:
# along-track distance, elevation, UTM easting, UTM northing, longitude, latitude.
ALONG_COL = 0
ELEVATION_COL = 1
EASTING_COL = 2
NORTHING_COL = 3
LONGITUDE_COL = 4
LATITUDE_COL = 5

# RANSAC parameters.
RANSAC_SIGMA_M = 0.3
RANSAC_ITERATIONS = 50
RANSAC_MIN_INLIERS = 10
RANSAC_MAX_ABS_SLOPE = 20.0

# Local direction estimation.
DIRECTION_STEP_M = 10.0
DIRECTION_SEARCH_RADIUS_M = 10.0

# Neighbor-distance feature.
K_NEIGHBORS = 20

# Direction-adaptive elliptical neighborhood.
ELLIPSE_SEMI_MAJOR_M = 10.0
ELLIPSE_SEMI_MINOR_M = 0.7
LOCAL_INDEX_RADIUS = 200

# Adaptive density threshold.
DENSITY_STEP_M = 3.0
DENSITY_SEARCH_RADIUS_M = 3.0
DENSITY_THRESHOLD_OFFSET = 0.1

# Similarity-weight function:
# exp[-(difference^2) * 0.125], equivalent to sigma = 2.
GAUSSIAN_EXPONENT_FACTOR = 0.125

# Local spatial-support filtering.
# Set ENABLE_CONTINUITY_FILTER = False to disable this final post-processing step.
ENABLE_CONTINUITY_FILTER = True
CONTINUITY_ALONG_WINDOW_M = 3.0
CONTINUITY_ELEVATION_WINDOW_M = 4.0
CONTINUITY_MIN_NEIGHBORS = 1

# Reproducibility:
# Set an integer (e.g., 42) to make RANSAC sampling repeatable.
# Keep None to retain non-fixed random sampling.
RANDOM_SEED = None

# Plot and output settings.
SAVE_PLOTS = True
SHOW_PLOTS = False
PLOT_DPI = 300


# =============================================================================
# 2. INPUT
# =============================================================================

def load_candidate_photons(file_path: Path) -> np.ndarray:
    """Load the six-column roughness-candidate TXT file and sort by distance."""
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Roughness candidate file not found:\n{file_path}\n"
            "Please check INPUT_FILE and run the roughness-screening module first."
        )

    data = np.loadtxt(
        file_path,
        delimiter=",",
        dtype=np.float64,
        ndmin=2
    )

    if data.shape[1] != 6:
        raise ValueError(
            "The roughness output must contain exactly 6 columns in this order:\n"
            "along-track distance, elevation, UTM easting, UTM northing, "
            "longitude, latitude.\n"
            f"Detected columns: {data.shape[1]}"
        )

    valid = np.all(np.isfinite(data), axis=1)
    n_invalid = int(np.count_nonzero(~valid))
    data = data[valid]

    if len(data) < K_NEIGHBORS:
        raise ValueError(
            f"Only {len(data)} valid photons were found. "
            f"At least {K_NEIGHBORS} are required for the {K_NEIGHBORS}-neighbor "
            "distance calculation."
        )

    data = data[np.argsort(data[:, ALONG_COL])]

    if n_invalid:
        print(f"Invalid rows removed: {n_invalid:,}")

    return data


# =============================================================================
# 3. RANSAC LINE FITTING
# =============================================================================

def fit_line_ransac_fast(
    points: np.ndarray,
    sigma: float = RANSAC_SIGMA_M,
    iters: int = RANSAC_ITERATIONS,
    min_inliers: int = RANSAC_MIN_INLIERS
):
    """Fit a local line using the same RANSAC logic as the original workflow."""
    points = np.asarray(points, dtype=np.float64)

    if len(points) < 2:
        return 0.0, 0.0, 0, np.empty((0, 2), dtype=np.float64)

    best_inliers = np.empty((0, 2), dtype=np.float64)
    best_a = 0.0
    best_b = 0.0

    n_points = len(points)

    for _ in range(iters):
        idx = np.random.choice(n_points, 2, replace=False)
        p1, p2 = points[idx]

        x1, y1 = p1
        x2, y2 = p2

        if abs(x2 - x1) < 1e-6:
            continue

        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1

        y_est = a * points[:, 0] + b
        mask = np.abs(y_est - points[:, 1]) < sigma
        inliers = points[mask]

        if (
            len(inliers) > len(best_inliers)
            and len(inliers) >= min_inliers
            and np.isfinite(a)
            and abs(a) < RANSAC_MAX_ABS_SLOPE
        ):
            best_inliers = inliers
            best_a = a
            best_b = b

    if len(best_inliers) == 0:
        return 0.0, 0.0, 0, np.empty((0, 2), dtype=np.float64)

    return best_a, best_b, len(best_inliers), best_inliers


# =============================================================================
# 4. LOCAL DIRECTION ESTIMATION
# =============================================================================

def estimate_local_directions(data: np.ndarray) -> np.ndarray:
    """
    Estimate local surface direction every 10 m and assign it to photons
    within the corresponding 10 m along-track interval.
    """
    along = data[:, ALONG_COL]
    angles = np.zeros(len(data), dtype=np.float64)

    nodes = np.arange(
        along[0],
        along[-1],
        DIRECTION_STEP_M
    )

    print("Estimating local directions using RANSAC...")

    for node in nodes:
        left = np.searchsorted(
            along,
            node - DIRECTION_SEARCH_RADIUS_M,
            side="left"
        )
        right = np.searchsorted(
            along,
            node + DIRECTION_SEARCH_RADIUS_M,
            side="left"
        )

        window = data[left:right]
        angle = 0.0

        if len(window) > 2:
            points = window[:, [ALONG_COL, ELEVATION_COL]]
            slope, _, _, inliers = fit_line_ransac_fast(points)

            if len(inliers) > 0:
                angle = math.atan(slope)

        share_left = np.searchsorted(along, node, side="left")
        share_right = np.searchsorted(
            along,
            node + DIRECTION_STEP_M,
            side="left"
        )
        angles[share_left:share_right] = angle

    print("Local direction estimation completed.")
    return angles


# =============================================================================
# 5. 20-NEIGHBOR DISTANCE FEATURE
# =============================================================================

def calculate_neighbor_distance(data: np.ndarray) -> np.ndarray:
    """
    Calculate the mean distance to the 20 nearest photons in the
    along-track/elevation plane. The target photon is included, matching
    the original KDTree query implementation.
    """
    points = data[:, [ALONG_COL, ELEVATION_COL]]
    tree = KDTree(points)

    distances, _ = tree.query(points, k=K_NEIGHBORS)
    mean_distance = np.mean(distances, axis=1)

    print("20-neighbor distance calculation completed.")
    return mean_distance


# =============================================================================
# 6. DIRECTION-ADAPTIVE SIMILARITY-WEIGHTED DENSITY
# =============================================================================

def calculate_similarity_weighted_density(
    data: np.ndarray,
    angles: np.ndarray,
    neighbor_distance: np.ndarray
) -> np.ndarray:
    """
    Calculate similarity-weighted density in a direction-adaptive elliptical
    neighborhood. The density is normalized by the ellipse area (pi * a * b).
    """
    n = len(data)
    along = data[:, ALONG_COL]
    elevation = data[:, ELEVATION_COL]

    density = np.zeros(n, dtype=np.float64)

    a = ELLIPSE_SEMI_MAJOR_M
    b = ELLIPSE_SEMI_MINOR_M
    ellipse_area = math.pi * a * b

    print("Calculating direction-adaptive similarity-weighted density...")

    for i in range(n):
        left = max(0, i - LOCAL_INDEX_RADIUS)
        right = min(n, i + LOCAL_INDEX_RADIUS)

        local_along = along[left:right]
        local_elevation = elevation[left:right]
        local_neighbor_distance = neighbor_distance[left:right]

        dx = along[i] - local_along
        dy = elevation[i] - local_elevation

        angle = angles[i]
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Rotate the local coordinate differences into the estimated
        # glacier-surface direction.
        x_rotated = cos_angle * dx + sin_angle * dy
        h_rotated = cos_angle * dy - sin_angle * dx

        # Elliptical neighborhood:
        # X^2/a^2 + H^2/b^2 <= 1.
        elliptical_distance = (
            (x_rotated * x_rotated) / (a * a)
            + (h_rotated * h_rotated) / (b * b)
        )

        inside = elliptical_distance <= 1.0
        local_values = local_neighbor_distance[inside]

        if local_values.size == 0:
            continue

        difference = neighbor_distance[i] - local_values

        # Original Gaussian similarity-weight function.
        weights = np.exp(
            -(difference * difference) * GAUSSIAN_EXPONENT_FACTOR
        )

        # Similarity-weighted density normalized by ellipse area.
        density[i] = np.sum(weights) / ellipse_area

    print("Density calculation completed.")
    print(
        f"Density range: {density.min():.6f} - {density.max():.6f}"
    )

    return density


# =============================================================================
# 7. ADAPTIVE DENSITY THRESHOLDING
# =============================================================================

def apply_adaptive_density_threshold(
    data: np.ndarray,
    density: np.ndarray
) -> np.ndarray:
    """
    Apply the original adaptive threshold in overlapping 3 m windows:
        threshold = max(mean(local density) - 0.1, 0)
    """
    along = data[:, ALONG_COL]

    # Internal column order:
    # along, elevation, density, easting, northing, longitude, latitude.
    full = np.column_stack((
        data[:, ALONG_COL],
        data[:, ELEVATION_COL],
        density,
        data[:, EASTING_COL],
        data[:, NORTHING_COL],
        data[:, LONGITUDE_COL],
        data[:, LATITUDE_COL]
    ))

    nodes = np.arange(
        along[0],
        along[-1],
        DENSITY_STEP_M
    )

    retained_segments = []

    print("Applying adaptive density threshold...")

    for node in nodes:
        left = np.searchsorted(
            along,
            node - DENSITY_SEARCH_RADIUS_M,
            side="left"
        )
        right = np.searchsorted(
            along,
            node + DENSITY_SEARCH_RADIUS_M,
            side="left"
        )

        window = full[left:right]

        if len(window) == 0:
            continue

        threshold = max(
            float(np.mean(window[:, 2])) - DENSITY_THRESHOLD_OFFSET,
            0.0
        )

        selected = window[window[:, 2] >= threshold]

        if len(selected):
            retained_segments.append(selected)

    if not retained_segments:
        raise RuntimeError(
            "No photons remained after adaptive density thresholding."
        )

    result = np.vstack(retained_segments)
    result = np.unique(result, axis=0)

    print(f"Photons after density thresholding: {len(result):,}")
    return result


# =============================================================================
# 8. LOCAL SPATIAL-SUPPORT FILTER
# =============================================================================

def remove_isolated_photons(result: np.ndarray):
    """
    Remove retained photons that have no other retained photon within the
    specified along-track/elevation neighborhood.

    A photon is retained if at least CONTINUITY_MIN_NEIGHBORS other photons
    occur within:
        |along-track difference| <= CONTINUITY_ALONG_WINDOW_M
        |elevation difference|    <= CONTINUITY_ELEVATION_WINDOW_M
    """
    if not ENABLE_CONTINUITY_FILTER or len(result) <= 1:
        return result, result[:0].copy()

    print("Applying local spatial-support filtering...")

    scaled_coordinates = np.column_stack((
        result[:, 0] / CONTINUITY_ALONG_WINDOW_M,
        result[:, 1] / CONTINUITY_ELEVATION_WINDOW_M
    ))

    tree = KDTree(scaled_coordinates)
    keep = np.zeros(len(result), dtype=bool)

    # The radius sqrt(2) includes the corners of the scaled rectangular
    # search window; exact rectangular limits are checked below.
    search_radius = math.sqrt(2.0)

    for i in range(len(result)):
        candidates = tree.query_radius(
            scaled_coordinates[i:i + 1],
            r=search_radius
        )[0]

        # Exclude the target photon itself.
        candidates = candidates[candidates != i]

        if candidates.size == 0:
            continue

        dx = np.abs(result[candidates, 0] - result[i, 0])
        dz = np.abs(result[candidates, 1] - result[i, 1])

        within_window = (
            (dx <= CONTINUITY_ALONG_WINDOW_M)
            & (dz <= CONTINUITY_ELEVATION_WINDOW_M)
        )

        neighbor_count = int(np.count_nonzero(within_window))

        if neighbor_count >= CONTINUITY_MIN_NEIGHBORS:
            keep[i] = True

    retained = result[keep]
    removed = result[~keep]

    print(f"Photons removed as isolated: {len(removed):,}")
    print(f"Photons after spatial-support filtering: {len(retained):,}")

    if len(retained) == 0:
        raise RuntimeError(
            "No photons remained after local spatial-support filtering. "
            "Consider checking the continuity-filter parameters."
        )

    return retained, removed


# =============================================================================
# 9. OUTPUT UTILITIES
# =============================================================================

def save_photon_txt(data: np.ndarray, output_path: Path):
    """Save photons in the same six-column order as the roughness output."""
    output_columns = [0, 1, 3, 4, 5, 6]

    np.savetxt(
        output_path,
        data[:, output_columns],
        delimiter=",",
        fmt="%.10f"
    )


def save_diagnostic_plots(
    input_data: np.ndarray,
    density: np.ndarray,
    denoised: np.ndarray,
    output_dir: Path,
    stem: str
):
    """Save overview and density plots."""
    plt.rcParams["font.family"] = "Times New Roman"

    # Plot 1: input candidate photons and retained signal photons.
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.scatter(
        input_data[:, ALONG_COL],
        input_data[:, ELEVATION_COL],
        s=4,
        color="gray",
        alpha=0.35,
        linewidths=0,
        label="Roughness candidate photons"
    )

    ax.scatter(
        denoised[:, 0],
        denoised[:, 1],
        s=8,
        linewidths=0,
        label="Retained signal photons"
    )

    ax.set_xlabel("Along-track distance (m)", fontsize=22)
    ax.set_ylabel("Elevation (m)", fontsize=22)
    ax.set_title("Denoising result", fontsize=22)
    ax.tick_params(
        axis="both",
        labelsize=18,
        direction="out",
        length=6,
        width=1.2
    )
    ax.legend(fontsize=15, loc="upper right")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.7)

    fig.tight_layout()
    fig.savefig(
        output_dir / f"{stem}_denoising_result.png",
        dpi=PLOT_DPI,
        bbox_inches="tight"
    )

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    # Plot 2: density distribution of input candidate photons.
    fig, ax = plt.subplots(figsize=(14, 5))

    scatter = ax.scatter(
        input_data[:, ALONG_COL],
        input_data[:, ELEVATION_COL],
        c=density,
        s=7,
        cmap="rainbow",
        linewidths=0
    )

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    colorbar.set_label("Similarity-weighted density", fontsize=18)
    colorbar.ax.tick_params(labelsize=15)

    ax.set_xlabel("Along-track distance (m)", fontsize=22)
    ax.set_ylabel("Elevation (m)", fontsize=22)
    ax.set_title("Direction-adaptive photon density", fontsize=22)
    ax.tick_params(
        axis="both",
        labelsize=18,
        direction="out",
        length=6,
        width=1.2
    )
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.7)

    fig.tight_layout()
    fig.savefig(
        output_dir / f"{stem}_density_distribution.png",
        dpi=PLOT_DPI,
        bbox_inches="tight"
    )

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    # Step 1: read roughness-screened candidate photons.
    data = load_candidate_photons(INPUT_FILE)
    stem = INPUT_FILE.stem

    print("\n===================================")
    print("Denoising input")
    print("===================================")
    print(f"Input file: {INPUT_FILE}")
    print(f"Input candidate photons: {len(data):,}")
    print(
        "Along-track range: "
        f"{data[:, ALONG_COL].min():.3f} - "
        f"{data[:, ALONG_COL].max():.3f} m"
    )
    print(
        "Elevation range: "
        f"{data[:, ELEVATION_COL].min():.3f} - "
        f"{data[:, ELEVATION_COL].max():.3f} m"
    )

    # Step 2: calculate the local 20-neighbor distance feature.
    neighbor_distance = calculate_neighbor_distance(data)

    # Step 3: estimate local glacier-surface directions.
    angles = estimate_local_directions(data)

    # Step 4: calculate direction-adaptive similarity-weighted density.
    density = calculate_similarity_weighted_density(
        data,
        angles,
        neighbor_distance
    )

    # Step 5: apply the adaptive density threshold.
    density_retained = apply_adaptive_density_threshold(data, density)

    # Step 6: apply the local spatial-support filter.
    denoised, isolated_removed = remove_isolated_photons(density_retained)

    # Step 7: save denoised photons in the six-column format required by
    # the subsequent fitting/crevasse-extraction module.
    denoised_path = OUTPUT_DIR / f"{stem}_denoised.txt"
    save_photon_txt(denoised, denoised_path)

    # Save isolated photons removed by the optional final filter for inspection.
    removed_path = OUTPUT_DIR / f"{stem}_isolated_photons_removed.txt"
    if len(isolated_removed):
        save_photon_txt(isolated_removed, removed_path)
    else:
        removed_path.write_text("", encoding="utf-8")

    # Save density values for diagnostics.
    density_path = OUTPUT_DIR / f"{stem}_density_values.csv"
    np.savetxt(
        density_path,
        np.column_stack((
            data[:, ALONG_COL],
            data[:, ELEVATION_COL],
            density
        )),
        delimiter=",",
        header="along_track_m,elevation_m,similarity_weighted_density",
        comments="",
        fmt="%.10f"
    )

    # Step 8: save diagnostic figures.
    if SAVE_PLOTS:
        save_diagnostic_plots(
            data,
            density,
            denoised,
            OUTPUT_DIR,
            stem
        )

    elapsed = time.perf_counter() - start_time

    print("\n========== Denoising completed ==========")
    print(f"Input candidate photons: {len(data):,}")
    print(f"After density thresholding: {len(density_retained):,}")
    print(f"Removed isolated photons: {len(isolated_removed):,}")
    print(f"Final denoised photons: {len(denoised):,}")
    print(
        "Final retention relative to roughness candidates: "
        f"{len(denoised) / len(data) * 100:.2f}%"
    )
    print(f"Denoised output: {denoised_path}")
    print(f"Removed isolated photons: {removed_path}")
    print(f"Density diagnostics: {density_path}")
    if SAVE_PLOTS:
        print(f"Diagnostic plots saved in: {OUTPUT_DIR}")
    print(f"Processing time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
