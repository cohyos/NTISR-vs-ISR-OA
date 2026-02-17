#!/usr/bin/env python3
"""
ISR Sensor Scan Parameter Optimizer

Systematically sweeps key ISR sensor and environmental parameters to find
the combination that maximizes detection performance.  For each point in
the parameter grid the script runs a reduced Monte Carlo simulation and
records detection rate, mean time-to-first-detection, and scan cycle time.

Results are written to a CSV file and a ranked summary is printed to the
console.  An optional heatmap PDF is generated when matplotlib is available.

Usage examples:
    python optimize_isr_scan.py                     # full sweep, 50 trials each
    python optimize_isr_scan.py --quick              # coarse grid for fast testing
    python optimize_isr_scan.py --trials 100         # more trials per point
    python optimize_isr_scan.py --output-dir ./opt   # custom output directory
    python optimize_isr_scan.py --top 20             # show top-20 results
"""

import argparse
import copy
import csv
import itertools
import os
import sys
import time

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Project imports — we rely on the existing simulation infrastructure.
# ---------------------------------------------------------------------------
from mc_engine import run_mc
from sensors import ISRLineScan

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config_default.yaml")
FT_TO_NM = 1.0 / 6076.12


# ===================================================================
# Configuration helpers
# ===================================================================

def load_base_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load the base YAML configuration."""
    if not os.path.isfile(path):
        sys.exit(f"ERROR: config file not found: {path}")
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def _deep_copy_cfg(cfg: dict) -> dict:
    """Return a deep copy of the configuration dict."""
    return copy.deepcopy(cfg)


# ===================================================================
# Parameter grid construction
# ===================================================================

def build_sweep_grid(quick: bool = False) -> dict:
    """
    Return a dict of parameter-name -> list-of-values for the sweep.

    When *quick* is True the grid is coarsened so a test run completes in
    a few minutes rather than hours.
    """
    if quick:
        return {
            "fov_deg":          [0.5, 1.0, 2.5, 5.0],
            "frame_rate_hz":    [10, 30, 80],
            "cell_radius_nm":   [0.5, 1.5],
            "slant_range_nm":   [5.0, 12.0],
            "altitude_ft":      [20000, 35000],
        }
    else:
        return {
            "fov_deg":          [0.2, 0.5, 1.0, 2.0, 3.0, 5.0],
            "frame_rate_hz":    [5, 10, 20, 30, 50, 100],
            "cell_radius_nm":   [0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
            "slant_range_nm":   [5.0, 8.0, 10.0, 14.0, 20.0],
            "altitude_ft":      [15000, 20000, 25000, 30000, 35000, 45000],
        }


def total_combinations(grid: dict) -> int:
    """Return the number of combinations in the parameter grid."""
    n = 1
    for values in grid.values():
        n *= len(values)
    return n


# ===================================================================
# Scan-cycle-time utility (no MC needed)
# ===================================================================

def compute_scan_cycle_time(altitude_ft: float, slant_range_nm: float,
                            cell_radius_nm: float, fov_deg: float,
                            frame_rate_hz: float) -> float:
    """
    Instantiate an ISRLineScan sensor and return its scan_cycle_time
    without running a full Monte Carlo simulation.

    Returns np.inf if the geometry is invalid (slant range <= altitude).
    """
    alt_nm = altitude_ft * FT_TO_NM
    if slant_range_nm <= alt_nm:
        return np.inf

    rng = np.random.default_rng(0)
    try:
        sensor = ISRLineScan(
            altitude_ft, slant_range_nm,
            0.0, 0.0,              # cell center at origin
            cell_radius_nm, 0.0,   # pd unused here
            rng, fov_deg, frame_rate_hz,
        )
        return sensor.scan_cycle_time
    except Exception:
        return np.inf


# ===================================================================
# Single-point evaluation
# ===================================================================

def evaluate_point(base_cfg: dict, params: dict, n_trials: int) -> dict:
    """
    Run a reduced MC simulation for one parameter combination and return
    a results dict with detection_rate, mean_ttfd, and scan_cycle_time.

    *params* keys: fov_deg, frame_rate_hz, cell_radius_nm,
                   slant_range_nm, altitude_ft.
    """
    cfg = _deep_copy_cfg(base_cfg)

    # Apply sweep parameters
    cfg["isr"]["fov_deg"]             = params["fov_deg"]
    cfg["isr"]["frame_rate_hz"]       = params["frame_rate_hz"]
    cfg["cell"]["radius_nm"]          = params["cell_radius_nm"]
    cfg["geometry"]["slant_range_nm"] = params["slant_range_nm"]
    cfg["platform"]["altitude_ft"]    = params["altitude_ft"]
    cfg["simulation"]["mc_trials"]    = n_trials

    # Validate geometry: slant range must exceed altitude
    alt_nm = params["altitude_ft"] * FT_TO_NM
    if params["slant_range_nm"] <= alt_nm:
        return {
            "detection_rate":  0.0,
            "mean_ttfd":       np.inf,
            "scan_cycle_time": np.inf,
            "skipped":         True,
            "skip_reason":     "slant_range <= altitude",
        }

    # Scan cycle time (cheap)
    sct = compute_scan_cycle_time(
        params["altitude_ft"], params["slant_range_nm"],
        params["cell_radius_nm"], params["fov_deg"],
        params["frame_rate_hz"],
    )

    # Run MC
    try:
        mc = run_mc(cfg, "isr")
        return {
            "detection_rate":  mc.overall_detect_fraction,
            "mean_ttfd":       mc.mean_ttfd,
            "scan_cycle_time": sct,
            "skipped":         False,
            "skip_reason":     "",
        }
    except Exception as exc:
        return {
            "detection_rate":  0.0,
            "mean_ttfd":       np.inf,
            "scan_cycle_time": sct,
            "skipped":         True,
            "skip_reason":     str(exc),
        }


# ===================================================================
# CSV I/O
# ===================================================================

CSV_COLUMNS = [
    "fov_deg",
    "frame_rate_hz",
    "cell_radius_nm",
    "slant_range_nm",
    "altitude_ft",
    "detection_rate_pct",
    "mean_ttfd_s",
    "scan_cycle_time_s",
    "skipped",
    "skip_reason",
]


def write_csv(rows: list[dict], path: str) -> None:
    """Write a list of result dicts to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ===================================================================
# Console summary
# ===================================================================

def print_summary(rows: list[dict], top_n: int = 10) -> None:
    """Print a ranked table of the top-N parameter sets to the console."""
    # Filter out skipped rows
    valid = [r for r in rows if not r.get("skipped", False)]
    if not valid:
        print("\nNo valid results to display (all parameter sets were skipped).")
        return

    # Sort: primary key = detection_rate_pct descending,
    #        secondary key = mean_ttfd_s ascending
    valid.sort(key=lambda r: (-r["detection_rate_pct"],
                                r["mean_ttfd_s"] if np.isfinite(r["mean_ttfd_s"]) else 1e9))

    top = valid[:top_n]

    hdr = (f"  {'Rank':>4s}  {'FOV':>6s}  {'FPS':>5s}  {'Rcell':>6s}  "
           f"{'SR':>6s}  {'Alt':>7s}  {'Det%':>7s}  {'TTFD':>8s}  {'Cycle':>8s}")
    units = (f"  {'':>4s}  {'[deg]':>6s}  {'[Hz]':>5s}  {'[nm]':>6s}  "
             f"{'[nm]':>6s}  {'[ft]':>7s}  {'':>7s}  {'[s]':>8s}  {'[s]':>8s}")
    sep = "  " + "-" * (len(hdr) - 2)

    print()
    print("=" * len(hdr))
    print(f"  TOP-{top_n} ISR PARAMETER SETS  (ranked by detection rate, then TTFD)")
    print("=" * len(hdr))
    print(hdr)
    print(units)
    print(sep)

    for idx, r in enumerate(top, start=1):
        ttfd_str = f"{r['mean_ttfd_s']:.1f}" if np.isfinite(r["mean_ttfd_s"]) else "N/A"
        cycle_str = f"{r['scan_cycle_time_s']:.2f}" if np.isfinite(r["scan_cycle_time_s"]) else "N/A"
        print(f"  {idx:>4d}  {r['fov_deg']:>6.2f}  {r['frame_rate_hz']:>5.0f}  "
              f"{r['cell_radius_nm']:>6.2f}  {r['slant_range_nm']:>6.1f}  "
              f"{r['altitude_ft']:>7.0f}  {r['detection_rate_pct']:>6.1f}%  "
              f"{ttfd_str:>8s}  {cycle_str:>8s}")

    print(sep)

    # Also print overall statistics
    det_rates = [r["detection_rate_pct"] for r in valid]
    print(f"\n  Total evaluated:  {len(valid)} valid parameter sets "
          f"({len(rows) - len(valid)} skipped)")
    print(f"  Detection rate range: {min(det_rates):.1f}% — {max(det_rates):.1f}%")
    if top:
        best = top[0]
        print(f"\n  BEST:  FOV={best['fov_deg']:.2f} deg, "
              f"FPS={best['frame_rate_hz']:.0f} Hz, "
              f"Rcell={best['cell_radius_nm']:.2f} nm, "
              f"SR={best['slant_range_nm']:.1f} nm, "
              f"Alt={best['altitude_ft']:.0f} ft  =>  "
              f"Det={best['detection_rate_pct']:.1f}%, "
              f"TTFD={best['mean_ttfd_s']:.1f}s")
    print()


# ===================================================================
# Heatmap generation (optional — requires matplotlib)
# ===================================================================

def generate_heatmap(rows: list[dict], output_path: str) -> bool:
    """
    Generate a multi-panel heatmap PDF showing detection rate as a function
    of FOV and frame rate, with one panel per (cell_radius, slant_range)
    combination at the best-performing altitude.

    Returns True if the PDF was written, False if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        return False

    valid = [r for r in rows if not r.get("skipped", False)]
    if not valid:
        return False

    # Determine unique parameter values
    fovs = sorted(set(r["fov_deg"] for r in valid))
    fpss = sorted(set(r["frame_rate_hz"] for r in valid))
    radii = sorted(set(r["cell_radius_nm"] for r in valid))
    srs = sorted(set(r["slant_range_nm"] for r in valid))

    # For each (radius, slant_range) pick the altitude that produced the
    # highest average detection rate across the FOV x FPS grid.
    altitudes = sorted(set(r["altitude_ft"] for r in valid))

    panels = []  # (radius, sr, altitude, 2-d array)
    for radius in radii:
        for sr in srs:
            best_alt = None
            best_mean_det = -1.0
            for alt in altitudes:
                subset = [r for r in valid
                          if r["cell_radius_nm"] == radius
                          and r["slant_range_nm"] == sr
                          and r["altitude_ft"] == alt]
                if not subset:
                    continue
                mean_det = np.mean([r["detection_rate_pct"] for r in subset])
                if mean_det > best_mean_det:
                    best_mean_det = mean_det
                    best_alt = alt

            if best_alt is None:
                continue

            grid = np.full((len(fpss), len(fovs)), np.nan)
            for r in valid:
                if (r["cell_radius_nm"] == radius
                        and r["slant_range_nm"] == sr
                        and r["altitude_ft"] == best_alt):
                    fi = fovs.index(r["fov_deg"])
                    ri = fpss.index(r["frame_rate_hz"])
                    grid[ri, fi] = r["detection_rate_pct"]

            if not np.all(np.isnan(grid)):
                panels.append((radius, sr, best_alt, grid))

    if not panels:
        return False

    n_panels = len(panels)
    ncols = min(n_panels, 3)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)

    vmin = min(r["detection_rate_pct"] for r in valid)
    vmax = max(r["detection_rate_pct"] for r in valid)
    norm = Normalize(vmin=max(vmin, 0), vmax=min(vmax, 100))

    for idx, (radius, sr, alt, grid) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        im = ax.imshow(grid, origin="lower", aspect="auto",
                        extent=[fovs[0], fovs[-1], fpss[0], fpss[-1]],
                        cmap="RdYlGn", norm=norm, interpolation="nearest")
        ax.set_xlabel("FOV [deg]")
        ax.set_ylabel("Frame Rate [Hz]")
        ax.set_title(f"R={radius} nm, SR={sr} nm\n(alt={alt:.0f} ft)",
                      fontsize=9)
        ax.set_xticks(fovs)
        ax.set_yticks(fpss)

    # Hide unused axes
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("ISR Detection Rate [%]  (FOV vs Frame Rate)", fontsize=12,
                 fontweight="bold")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.04)
    cbar.set_label("Detection Rate [%]")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return True


# ===================================================================
# Progress display
# ===================================================================

def format_eta(seconds: float) -> str:
    """Format an ETA in seconds to a human-readable string."""
    if not np.isfinite(seconds) or seconds < 0:
        return "???"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep ISR sensor parameters to find the combination that "
            "maximizes detection performance."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python optimize_isr_scan.py --quick\n"
            "  python optimize_isr_scan.py --trials 100 --top 20\n"
            "  python optimize_isr_scan.py --output-dir ./opt --no-heatmap\n"
        ),
    )
    parser.add_argument(
        "--trials", type=int, default=50, metavar="N",
        help="Number of MC trials per parameter combination (default: 50)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results", metavar="DIR",
        help="Directory for output files (default: ./results)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use a coarser parameter grid for fast testing",
    )
    parser.add_argument(
        "--top", type=int, default=10, metavar="N",
        help="Number of top results to display in the summary (default: 10)",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="Skip heatmap PDF generation even if matplotlib is available",
    )
    parser.add_argument(
        "--config", type=str, default=None, metavar="YAML",
        help="Path to base config YAML (default: config_default.yaml)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load base configuration
    # ------------------------------------------------------------------
    config_path = args.config if args.config else DEFAULT_CONFIG_PATH
    base_cfg = load_base_config(config_path)

    # ------------------------------------------------------------------
    # Build parameter grid
    # ------------------------------------------------------------------
    grid = build_sweep_grid(quick=args.quick)
    n_total = total_combinations(grid)

    param_names = list(grid.keys())
    param_values = [grid[k] for k in param_names]

    print("=" * 70)
    print("  ISR Scan Parameter Optimizer")
    print("=" * 70)
    print(f"  Base config : {os.path.abspath(config_path)}")
    print(f"  MC trials   : {args.trials} per combination")
    print(f"  Grid mode   : {'quick (coarse)' if args.quick else 'full'}")
    print(f"  Combinations: {n_total}")
    print()
    for name in param_names:
        vals = grid[name]
        print(f"    {name:20s}: {vals}")
    print()
    print(f"  Output dir  : {os.path.abspath(args.output_dir)}")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Run sweep
    # ------------------------------------------------------------------
    all_rows: list[dict] = []
    t_global_start = time.time()
    completed = 0
    skipped_count = 0

    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))
        completed += 1

        # Progress header
        elapsed = time.time() - t_global_start
        if completed > 1:
            avg_per = elapsed / (completed - 1)
            eta = avg_per * (n_total - completed + 1)
        else:
            eta = np.inf

        label = (f"FOV={params['fov_deg']:.1f}  FPS={params['frame_rate_hz']:.0f}  "
                 f"R={params['cell_radius_nm']:.1f}  SR={params['slant_range_nm']:.0f}  "
                 f"Alt={params['altitude_ft']:.0f}")
        print(f"[{completed}/{n_total}]  {label}  "
              f"(elapsed {format_eta(elapsed)}, ETA {format_eta(eta)})",
              end="", flush=True)

        result = evaluate_point(base_cfg, params, args.trials)

        row = {
            "fov_deg":            params["fov_deg"],
            "frame_rate_hz":      params["frame_rate_hz"],
            "cell_radius_nm":     params["cell_radius_nm"],
            "slant_range_nm":     params["slant_range_nm"],
            "altitude_ft":        params["altitude_ft"],
            "detection_rate_pct": round(result["detection_rate"] * 100, 2),
            "mean_ttfd_s":        round(result["mean_ttfd"], 3)
                                  if np.isfinite(result["mean_ttfd"]) else float("inf"),
            "scan_cycle_time_s":  round(result["scan_cycle_time"], 4)
                                  if np.isfinite(result["scan_cycle_time"]) else float("inf"),
            "skipped":            result["skipped"],
            "skip_reason":        result.get("skip_reason", ""),
        }
        all_rows.append(row)

        if result["skipped"]:
            skipped_count += 1
            print(f"  => SKIPPED ({result['skip_reason']})")
        else:
            det_str = f"{row['detection_rate_pct']:.1f}%"
            ttfd_str = (f"{row['mean_ttfd_s']:.1f}s"
                        if np.isfinite(row["mean_ttfd_s"]) else "N/A")
            print(f"  => Det={det_str}, TTFD={ttfd_str}")

    total_elapsed = time.time() - t_global_start

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "isr_optimization.csv")
    write_csv(all_rows, csv_path)
    print(f"\nCSV results written to: {os.path.abspath(csv_path)}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print_summary(all_rows, top_n=args.top)

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------
    if not args.no_heatmap:
        heatmap_path = os.path.join(args.output_dir, "isr_optimization_heatmap.pdf")
        print("Generating heatmap PDF ...", end=" ", flush=True)
        if generate_heatmap(all_rows, heatmap_path):
            print(f"saved to {os.path.abspath(heatmap_path)}")
        else:
            print("skipped (matplotlib not available or no valid data)")

    # ------------------------------------------------------------------
    # Final timing
    # ------------------------------------------------------------------
    print(f"\nTotal wall-clock time: {format_eta(total_elapsed)}  "
          f"({completed} combinations, {skipped_count} skipped)")


if __name__ == "__main__":
    main()
