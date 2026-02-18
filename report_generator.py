#!/usr/bin/env python3
"""
Comprehensive Report Generator for NTISR vs ISR Operational Analysis.

Produces a multi-page PDF report that serves as both documentation and
results archive.  The report includes:

  1. Methodology overview (how the simulation works, what it compares)
  2. Sensor model descriptions (ISR, NTISR S&S, NTISR FMV)
  3. Detection and target models
  4. Geometry and footprint calculations
  5. Current configuration summary
  6. ISR optimisation results (parameter sweep with CSV export)
  7. Sensitivity analysis (one-at-a-time parameter sweeps)

The generator is designed to be called from the interactive menu or
directly from the command line.
"""

import copy
import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mc_engine import run_mc
from sensors import ISRLineScan
from geometry import ground_footprint_at_cell, NM_TO_FT, FT_TO_NM

# =====================================================================
# Text content for the manual / methodology sections
# =====================================================================

_TITLE = "NTISR vs ISR Operational Analysis\nComprehensive Report"

_METHODOLOGY_TEXT = """\
Methodology Overview

This tool compares the detection performance of three airborne EO/IR
sensor search modes against a single moving ground target (car) inside
a circular geographic cell.

Monte Carlo Approach
  For each sensor mode the simulation runs N independent random trials.
  In every trial a car is placed at a random position inside the cell
  with a random heading.  A sensor object sweeps the cell according to
  its mode-specific scan logic.  At each simulation time step:
    1. The car advances (constant speed, periodic random heading changes,
       reflective boundary at the cell edge).
    2. The sensor updates its pointing / footprint position.
    3. A detection check is performed: if the car falls inside the
       sensor footprint the mode's probabilistic detection model is
       evaluated.
  After all time steps the trial records time-to-first-detection (TTFD)
  and total detection events.  Aggregate statistics (mean/median TTFD,
  cumulative detection probability, overall detection rate) are computed
  across all trials.

Platform Model
  The platform is treated as stationary (orbiting) above the cell for
  the full simulation duration.  Platform speed and heading affect only
  the display/reporting and do NOT alter slant-range or viewing geometry
  over time.

Comparison Framework
  The three sensor modes are evaluated under identical conditions
  (same platform, same cell, same target behaviour, same detection
  model parameters).  The key performance metrics are:
    - Detection rate: fraction of trials where the target was detected
    - Mean / median TTFD: how quickly the sensor finds the target
    - Detection event count: how many times the target is re-detected\
"""

_SENSOR_ISR_TEXT = """\
Sensor Mode 1: ISR Line Scanner (Back-Scan Mirror)

The ISR sensor has a fixed FOV (default 1.0 deg).  It steps through a
boustrophedon (row-by-row lawnmower) raster covering the full circular
cell at the configured frame rate (default 30 Hz).

The back-scan mirror stabilises the line-of-sight during each frame so
the image is smear-free, but it does NOT widen the instantaneous field
-- the sensor sees exactly one FOV-sized patch per frame.

Scan Pattern
  Positions are laid out on a square grid with 20% overlap between
  adjacent frames.  Only positions whose centres fall inside the
  circular cell are included.  Odd-numbered rows are reversed to
  create the boustrophedon (serpentine) path.

Detection Model
  Each frame where the target is inside the FOV is an independent
  detection opportunity.  A single uniform random draw is compared
  against pd_in_fov -- if the draw is less, the target is detected.
  No minimum dwell is required; even a single 1/30 s frame suffices.

Key Parameters
  - fov_deg: angular width of one frame (larger = faster coverage but
    lower resolution)
  - frame_rate_hz: how many frames per second (higher = faster raster
    but shorter exposure per frame)
  - scan_cycle_time: time to complete one full raster pass of the cell
    (derived from the number of raster positions and frame rate)\
"""

_SENSOR_SS_TEXT = """\
Sensor Mode 2: NTISR Step-and-Stare

The narrow-FOV targeting pod is pointed at one position, dwells
(stares) to build up detection probability, then slews (rotates) to
the next position in a systematic raster pattern.

Scan Pattern
  Same boustrophedon grid as ISR, but with the NTISR FOV (default
  0.5 deg), 20% overlap.  The pod starts at a random position in the
  grid to avoid bias from always beginning at the cell edge.

Timing
  At each position the sensor dwells for dwell_time_s (default 2.0 s).
  After the dwell it calculates the angular distance to the next grid
  position and slews at slew_rate_deg_s (default 20 deg/s).  During
  the slew the footprint size is set to zero (no detection possible).

Detection Model
  The target must be inside the FOV.  After the minimum dwell threshold
  (min_dwell_for_detect_s, default 0.5 s) the detection probability
  accumulates each time step:
    p_step = 1 - (1 - pd_in_fov) ^ (dt / effective_dwell)
  where effective_dwell = dwell_time_s - min_dwell_s.  This is
  calibrated so that one complete dwell gives a cumulative detection
  probability equal to pd_in_fov.

Key Parameters
  - fov_deg, dwell_time_s, slew_rate_deg_s, min_dwell_for_detect_s\
"""

_SENSOR_FMV_TEXT = """\
Sensor Mode 3: NTISR Full Motion Video (FMV) Search

An operator watches a live video feed from the targeting pod and
manually pans across the cell, biased toward roads and linear features
where vehicles are likely to be found.

Operator Model
  The operator scan is modelled as a biased random walk:
  - A random heading is drawn periodically (exponential interval,
    mean ~4 s).
  - The heading is blended with an attraction vector toward the
    nearest road feature, weighted by road_bias (default 0.7).
  - The pod moves at scan_speed = slew_rate * ground_range *
    search_speed_factor.
  - When the footprint reaches the cell boundary the operator
    reflects inward with a small random jitter.

Road Network
  Each trial generates 2-5 random linear "road" features through the
  cell.  These serve as attractors for the operator's scan path.

Detection Model
  The operator must keep the target continuously in the FOV for a
  recognition time (min_dwell_s, default 0.3 s).  After that, Pd
  accumulates per time step over a 2 s recognition window:
    p_step = 1 - (1 - pd_in_fov) ^ (dt / 2.0)

Key Parameters
  - fov_deg, slew_rate_deg_s, search_speed_factor, road_bias,
    revisit_tendency\
"""

_TARGET_MODEL_TEXT = """\
Target (Car) Model

The ground target is a vehicle moving inside the circular search cell.

  - Random initial position (uniform distribution inside the circle).
  - Random initial heading (0-2pi).
  - Constant speed (default 30 kts = 0.00833 nm/s).
  - Periodic random heading changes with exponentially distributed
    intervals (mean = heading_change_interval_s, default 15 s).
  - Reflective boundary: when the car hits the cell edge, the velocity
    component normal to the boundary is reversed, keeping the car
    inside the cell at all times.\
"""

_GEOMETRY_TEXT = """\
Geometry & Footprint Calculations

Platform-to-ground geometry:
  - altitude_ft: platform height above ground level
  - slant_range_nm: line-of-sight distance from platform to cell centre
  - ground_range_nm = sqrt(slant_range^2 - altitude^2)
  - depression_angle = arctan(altitude / ground_range)

FOV to ground footprint:
  footprint_nm = slant_range * fov_rad / cos(depression_angle)
  This is the small-angle approximation.  Larger depression angles
  (more overhead) shrink the footprint.

Constraint: slant_range_nm must be greater than altitude in nm,
otherwise the geometry is physically impossible (the platform would
be above the target at a steeper angle than vertical).\
"""


# =====================================================================
# Helper: render a text page in the PDF
# =====================================================================

def _text_page(pdf, title, body, fontsize=9):
    """Add a page with a title and body text to the PDF."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.05, 0.97, title, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")
    ax.text(0.05, 0.91, body, transform=ax.transAxes,
            fontsize=fontsize, va="top", fontfamily="monospace",
            linespacing=1.35)
    pdf.savefig(fig)
    plt.close(fig)


# =====================================================================
# Sensitivity analysis
# =====================================================================

def _run_sensitivity(cfg, param_path, param_label, values, mode="isr",
                     trials=50, progress_fn=None):
    """
    Run MC simulations varying one parameter at a time.

    Args:
        cfg: base config dict (will be deep-copied)
        param_path: tuple of keys to reach the parameter, e.g.
                    ('isr', 'fov_deg')
        param_label: human-readable label for the parameter
        values: list of parameter values to sweep
        mode: sensor mode to test
        trials: MC trials per point
        progress_fn: optional callable(i, n, label)

    Returns:
        list of dicts with keys: value, detection_rate, mean_ttfd
    """
    results = []
    for i, val in enumerate(values):
        test_cfg = copy.deepcopy(cfg)
        # Navigate to the parameter
        node = test_cfg
        for key in param_path[:-1]:
            node = node[key]
        node[param_path[-1]] = val
        test_cfg["simulation"]["mc_trials"] = trials

        # Validate geometry
        alt_nm = test_cfg["platform"]["altitude_ft"] * FT_TO_NM
        if test_cfg["geometry"]["slant_range_nm"] <= alt_nm:
            results.append({"value": val, "detection_rate": 0.0,
                            "mean_ttfd": np.inf})
            continue

        if progress_fn:
            progress_fn(i + 1, len(values), f"{param_label}={val}")

        try:
            mc = run_mc(test_cfg, mode)
            results.append({
                "value": val,
                "detection_rate": mc.overall_detect_fraction * 100,
                "mean_ttfd": mc.mean_ttfd,
            })
        except Exception:
            results.append({"value": val, "detection_rate": 0.0,
                            "mean_ttfd": np.inf})
    return results


def _plot_sensitivity(pdf, param_label, unit, values, det_rates,
                      mean_ttfds, mode_label="ISR"):
    """Plot a two-panel sensitivity chart and add it to the PDF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    color1 = "#4472C4"
    color2 = "#ED7D31"

    ax1.plot(values, det_rates, "o-", color=color1, linewidth=2,
             markersize=6)
    ax1.set_xlabel(f"{param_label} [{unit}]", fontsize=11)
    ax1.set_ylabel("Detection Rate [%]", fontsize=11, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_title(f"Detection Rate vs {param_label}", fontsize=12,
                  fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    finite_ttfds = [t if np.isfinite(t) else None for t in mean_ttfds]
    valid_x = [v for v, t in zip(values, finite_ttfds) if t is not None]
    valid_t = [t for t in finite_ttfds if t is not None]

    if valid_t:
        ax2.plot(valid_x, valid_t, "s-", color=color2, linewidth=2,
                 markersize=6)
    ax2.set_xlabel(f"{param_label} [{unit}]", fontsize=11)
    ax2.set_ylabel("Mean TTFD [s]", fontsize=11, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_title(f"Mean TTFD vs {param_label}", fontsize=12,
                  fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    fig.suptitle(f"Sensitivity Analysis: {param_label} ({mode_label})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# =====================================================================
# ISR Optimisation (embedded, lighter version)
# =====================================================================

def _run_optimisation(cfg, quick=True, trials=50, progress_fn=None):
    """
    Run the ISR parameter sweep and return a list of result dicts.

    The ISR FOV is fixed from the config and is NOT swept.
    """
    import itertools

    if quick:
        grid = {
            "frame_rate_hz":  [10, 30, 80],
            "cell_radius_nm": [0.5, 1.0, 1.5],
            "slant_range_nm": [5.0, 10.0],
            "altitude_ft":    [20000, 35000],
        }
    else:
        grid = {
            "frame_rate_hz":  [5, 10, 20, 30, 50, 100],
            "cell_radius_nm": [0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
            "slant_range_nm": [5.0, 8.0, 10.0, 14.0, 20.0],
            "altitude_ft":    [15000, 20000, 25000, 30000, 35000, 45000],
        }

    param_names = list(grid.keys())
    param_values = [grid[k] for k in param_names]
    n_total = 1
    for v in param_values:
        n_total *= len(v)

    rows = []
    completed = 0

    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))
        completed += 1

        if progress_fn:
            progress_fn(completed, n_total,
                        f"FPS={params['frame_rate_hz']:.0f} "
                        f"R={params['cell_radius_nm']:.1f}")

        test_cfg = copy.deepcopy(cfg)
        # FOV stays at config default
        test_cfg["isr"]["frame_rate_hz"] = params["frame_rate_hz"]
        test_cfg["cell"]["radius_nm"] = params["cell_radius_nm"]
        test_cfg["geometry"]["slant_range_nm"] = params["slant_range_nm"]
        test_cfg["platform"]["altitude_ft"] = params["altitude_ft"]
        test_cfg["simulation"]["mc_trials"] = trials

        fov_deg = test_cfg["isr"]["fov_deg"]

        alt_nm = params["altitude_ft"] * FT_TO_NM
        if params["slant_range_nm"] <= alt_nm:
            rows.append({
                "frame_rate_hz": params["frame_rate_hz"],
                "cell_radius_nm": params["cell_radius_nm"],
                "slant_range_nm": params["slant_range_nm"],
                "altitude_ft": params["altitude_ft"],
                "detection_rate_pct": 0.0,
                "mean_ttfd_s": float("inf"),
                "scan_cycle_time_s": float("inf"),
                "skipped": True,
                "skip_reason": "slant_range <= altitude",
            })
            continue

        # Scan cycle time
        try:
            rng = np.random.default_rng(0)
            sensor = ISRLineScan(
                params["altitude_ft"], params["slant_range_nm"],
                0.0, 0.0, params["cell_radius_nm"], 0.0, rng,
                fov_deg, params["frame_rate_hz"])
            sct = sensor.scan_cycle_time
        except Exception:
            sct = float("inf")

        try:
            mc = run_mc(test_cfg, "isr")
            rows.append({
                "frame_rate_hz": params["frame_rate_hz"],
                "cell_radius_nm": params["cell_radius_nm"],
                "slant_range_nm": params["slant_range_nm"],
                "altitude_ft": params["altitude_ft"],
                "detection_rate_pct": round(mc.overall_detect_fraction * 100, 2),
                "mean_ttfd_s": round(mc.mean_ttfd, 3) if np.isfinite(mc.mean_ttfd) else float("inf"),
                "scan_cycle_time_s": round(sct, 4) if np.isfinite(sct) else float("inf"),
                "skipped": False,
                "skip_reason": "",
            })
        except Exception as exc:
            rows.append({
                "frame_rate_hz": params["frame_rate_hz"],
                "cell_radius_nm": params["cell_radius_nm"],
                "slant_range_nm": params["slant_range_nm"],
                "altitude_ft": params["altitude_ft"],
                "detection_rate_pct": 0.0,
                "mean_ttfd_s": float("inf"),
                "scan_cycle_time_s": round(sct, 4) if np.isfinite(sct) else float("inf"),
                "skipped": True,
                "skip_reason": str(exc),
            })

    return rows


def _write_optimisation_csv(rows, path):
    """Write optimisation results to CSV."""
    columns = [
        "frame_rate_hz", "cell_radius_nm", "slant_range_nm",
        "altitude_ft", "detection_rate_pct", "mean_ttfd_s",
        "scan_cycle_time_s", "skipped", "skip_reason",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _optimisation_summary_page(pdf, rows, top_n=15):
    """Create a page showing the top-N optimisation results as a table."""
    valid = [r for r in rows if not r.get("skipped", False)]
    if not valid:
        _text_page(pdf, "ISR Optimisation Results",
                   "No valid results (all parameter sets were skipped).")
        return

    valid.sort(key=lambda r: (
        -r["detection_rate_pct"],
        r["mean_ttfd_s"] if np.isfinite(r["mean_ttfd_s"]) else 1e9))

    top = valid[:top_n]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.97,
            f"ISR Optimisation — Top {min(top_n, len(top))} Parameter Sets",
            transform=ax.transAxes, fontsize=14, ha="center",
            fontweight="bold", va="top")

    col_labels = ["Rank", "FPS\n[Hz]", "Cell R\n[nm]",
                  "SR\n[nm]", "Alt\n[ft]", "Det\n[%]", "TTFD\n[s]",
                  "Cycle\n[s]"]

    table_data = []
    for i, r in enumerate(top, 1):
        ttfd = f"{r['mean_ttfd_s']:.1f}" if np.isfinite(r["mean_ttfd_s"]) else "N/A"
        cycle = f"{r['scan_cycle_time_s']:.2f}" if np.isfinite(r["scan_cycle_time_s"]) else "N/A"
        table_data.append([
            str(i),
            f"{r['frame_rate_hz']:.0f}",
            f"{r['cell_radius_nm']:.2f}",
            f"{r['slant_range_nm']:.1f}",
            f"{r['altitude_ft']:.0f}",
            f"{r['detection_rate_pct']:.1f}",
            ttfd,
            cycle,
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    for (row_i, col_i), cell in table.get_celld().items():
        if row_i == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_i % 2 == 0:
            cell.set_facecolor("#D6E4F0")

    # Stats below table
    det_rates = [r["detection_rate_pct"] for r in valid]
    stats = (f"Total evaluated: {len(valid)} valid "
             f"({len(rows) - len(valid)} skipped)   |   "
             f"Detection rate range: {min(det_rates):.1f}% - "
             f"{max(det_rates):.1f}%")
    ax.text(0.5, 0.03, stats, transform=ax.transAxes,
            fontsize=9, ha="center", style="italic")

    pdf.savefig(fig)
    plt.close(fig)


def _optimisation_heatmap_page(pdf, rows):
    """Create heatmap page(s) showing detection rate vs frame rate and cell radius."""
    from matplotlib.colors import Normalize

    valid = [r for r in rows if not r.get("skipped", False)]
    if not valid:
        return

    fpss = sorted(set(r["frame_rate_hz"] for r in valid))
    radii = sorted(set(r["cell_radius_nm"] for r in valid))
    srs = sorted(set(r["slant_range_nm"] for r in valid))
    altitudes = sorted(set(r["altitude_ft"] for r in valid))

    panels = []
    for sr in srs:
        best_alt, best_mean = None, -1.0
        for alt in altitudes:
            subset = [r for r in valid
                      if r["slant_range_nm"] == sr
                      and r["altitude_ft"] == alt]
            if not subset:
                continue
            mean_det = np.mean([r["detection_rate_pct"] for r in subset])
            if mean_det > best_mean:
                best_mean = mean_det
                best_alt = alt

        if best_alt is None:
            continue

        grid = np.full((len(fpss), len(radii)), np.nan)
        for r in valid:
            if (r["slant_range_nm"] == sr
                    and r["altitude_ft"] == best_alt):
                ri = radii.index(r["cell_radius_nm"])
                fi = fpss.index(r["frame_rate_hz"])
                grid[fi, ri] = r["detection_rate_pct"]

        if not np.all(np.isnan(grid)):
            panels.append((sr, best_alt, grid))

    if not panels:
        return

    n_panels = len(panels)
    ncols = min(n_panels, 3)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)

    vmin = min(r["detection_rate_pct"] for r in valid)
    vmax = max(r["detection_rate_pct"] for r in valid)
    norm = Normalize(vmin=max(vmin, 0), vmax=min(vmax, 100))

    for idx, (sr, alt, grid) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        im = ax.imshow(grid, origin="lower", aspect="auto",
                       extent=[radii[0], radii[-1], fpss[0], fpss[-1]],
                       cmap="RdYlGn", norm=norm, interpolation="nearest")
        ax.set_xlabel("Cell Radius [nm]")
        ax.set_ylabel("Frame Rate [Hz]")
        ax.set_title(f"SR={sr} nm (alt={alt:.0f} ft)",
                     fontsize=9)
        ax.set_xticks(radii)
        ax.set_yticks(fpss)

    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("ISR Detection Rate [%]  (Frame Rate vs Cell Radius, FOV fixed)",
                 fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.04)
    cbar.set_label("Detection Rate [%]")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Configuration summary page
# =====================================================================

def _config_summary_page(pdf, cfg):
    """Create a page with the full configuration summary."""
    alt = cfg["platform"]["altitude_ft"]
    sr = cfg["geometry"]["slant_range_nm"]

    lines = []
    lines.append("Platform")
    lines.append(f"  Altitude:        {alt:.0f} ft AGL")
    lines.append(f"  Speed:           {cfg['platform']['speed_kts']} kts")
    lines.append(f"  Heading:         {cfg['platform']['heading_deg']} deg")
    lines.append(f"  Slant Range:     {sr} nm")
    lines.append("")
    lines.append("Cell & Target")
    lines.append(f"  Cell Radius:     {cfg['cell']['radius_nm']} nm "
                 f"({cfg['cell']['radius_nm'] * NM_TO_FT:.0f} ft)")
    lines.append(f"  Car Speed:       {cfg['car']['speed_kts']} kts")
    lines.append(f"  Heading Chg Int: {cfg['car']['heading_change_interval_s']} s")
    lines.append("")
    lines.append("ISR Line Scanner")
    lines.append(f"  FOV:             {cfg['isr']['fov_deg']} deg")
    lines.append(f"  Frame Rate:      {cfg['isr']['frame_rate_hz']} Hz")
    fp_isr = ground_footprint_at_cell(alt, sr, cfg['isr']['fov_deg'])
    lines.append(f"  Footprint:       {fp_isr:.4f} nm ({fp_isr * NM_TO_FT:.0f} ft)")
    lines.append("")
    lines.append("NTISR Step-and-Stare")
    lines.append(f"  FOV:             {cfg['ntisr_step_stare']['fov_deg']} deg")
    lines.append(f"  Dwell Time:      {cfg['ntisr_step_stare']['dwell_time_s']} s")
    lines.append(f"  Slew Rate:       {cfg['ntisr_step_stare']['slew_rate_deg_s']} deg/s")
    lines.append(f"  Scan Pattern:    {cfg['ntisr_step_stare']['scan_pattern']}")
    fp_ss = ground_footprint_at_cell(alt, sr, cfg['ntisr_step_stare']['fov_deg'])
    lines.append(f"  Footprint:       {fp_ss:.4f} nm ({fp_ss * NM_TO_FT:.0f} ft)")
    lines.append("")
    lines.append("NTISR FMV Search")
    lines.append(f"  FOV:             {cfg['ntisr_fmv']['fov_deg']} deg")
    lines.append(f"  Slew Rate:       {cfg['ntisr_fmv']['slew_rate_deg_s']} deg/s")
    lines.append(f"  Speed Factor:    {cfg['ntisr_fmv']['search_speed_factor']}")
    lines.append(f"  Road Bias:       {cfg['ntisr_fmv']['road_bias']}")
    lines.append(f"  Revisit Tend.:   {cfg['ntisr_fmv']['revisit_tendency']}")
    fp_fmv = ground_footprint_at_cell(alt, sr, cfg['ntisr_fmv']['fov_deg'])
    lines.append(f"  Footprint:       {fp_fmv:.4f} nm ({fp_fmv * NM_TO_FT:.0f} ft)")
    lines.append("")
    lines.append("Detection Model")
    lines.append(f"  Pd in FOV:       {cfg['detection']['pd_in_fov']}")
    lines.append(f"  Decay Exponent:  {cfg['detection']['pd_decay_exponent']}")
    lines.append(f"  Min Dwell:       {cfg['detection']['min_dwell_for_detect_s']} s")
    lines.append("")
    lines.append("Simulation")
    lines.append(f"  Duration:        {cfg['simulation']['duration_s']} s")
    lines.append(f"  Time Step:       {cfg['simulation']['time_step_s']} s")
    lines.append(f"  MC Trials:       {cfg['simulation']['mc_trials']}")
    lines.append(f"  Random Seed:     {cfg['simulation'].get('random_seed', 'None')}")

    _text_page(pdf, "Configuration Summary", "\n".join(lines), fontsize=9)


# =====================================================================
# Main report generator
# =====================================================================

def generate_comprehensive_report(cfg, output_dir="./results",
                                  run_optimization=True,
                                  optimization_quick=True,
                                  optimization_trials=50,
                                  sensitivity_trials=50,
                                  progress_fn=None):
    """
    Generate a comprehensive PDF report with manual, optimisation
    results, CSV export, and sensitivity analysis.

    Args:
        cfg: configuration dict
        output_dir: directory for all outputs
        run_optimization: whether to run the ISR parameter sweep
        optimization_quick: use coarse grid for speed
        optimization_trials: MC trials per optimisation point
        sensitivity_trials: MC trials per sensitivity point
        progress_fn: optional callable(message) for status updates

    Returns:
        dict with paths to generated files
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "comprehensive_report.pdf")
    csv_path = os.path.join(output_dir, "isr_optimization.csv")
    sensitivity_csv_path = os.path.join(output_dir, "sensitivity_analysis.csv")

    def _log(msg):
        if progress_fn:
            progress_fn(msg)
        print(msg)

    _log("Generating comprehensive report...")
    t_start = time.time()

    with PdfPages(pdf_path) as pdf:
        # ── Title page ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.75, _TITLE, transform=ax.transAxes,
                fontsize=20, ha="center", va="top", fontweight="bold",
                linespacing=1.5)
        import datetime
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        ax.text(0.5, 0.60, f"Generated: {date_str}",
                transform=ax.transAxes, fontsize=11, ha="center",
                style="italic")
        ax.text(0.5, 0.45, "Table of Contents", transform=ax.transAxes,
                fontsize=13, ha="center", fontweight="bold")
        toc = [
            "1.  Methodology Overview",
            "2.  Sensor Mode 1: ISR Line Scanner",
            "3.  Sensor Mode 2: NTISR Step-and-Stare",
            "4.  Sensor Mode 3: NTISR FMV Search",
            "5.  Target (Car) Model",
            "6.  Geometry & Footprint Calculations",
            "7.  Configuration Summary",
        ]
        if run_optimization:
            toc.append("8.  ISR Optimisation Results")
            toc.append("9.  ISR Optimisation Heatmap")
            toc.append("10. Sensitivity Analysis")
        else:
            toc.append("8.  Sensitivity Analysis")
        ax.text(0.15, 0.38, "\n".join(toc), transform=ax.transAxes,
                fontsize=10, va="top", fontfamily="monospace",
                linespacing=1.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ── Manual / methodology pages ──────────────────────────────
        _log("  Writing methodology sections...")
        _text_page(pdf, "1. Methodology Overview", _METHODOLOGY_TEXT)
        _text_page(pdf, "2. ISR Line Scanner (Back-Scan Mirror)",
                   _SENSOR_ISR_TEXT)
        _text_page(pdf, "3. NTISR Step-and-Stare", _SENSOR_SS_TEXT)
        _text_page(pdf, "4. NTISR FMV Search", _SENSOR_FMV_TEXT)
        _text_page(pdf, "5. Target (Car) Model", _TARGET_MODEL_TEXT)
        _text_page(pdf, "6. Geometry & Footprint Calculations",
                   _GEOMETRY_TEXT)

        # ── Configuration summary ───────────────────────────────────
        _config_summary_page(pdf, cfg)

        # ── ISR Optimisation ────────────────────────────────────────
        opt_rows = None
        if run_optimization:
            grid_label = "quick (coarse)" if optimization_quick else "full"
            _log(f"  Running ISR optimisation ({grid_label} grid, "
                 f"{optimization_trials} trials/point)...")

            def _opt_progress(i, n, label):
                if i % max(1, n // 20) == 0 or i == n:
                    _log(f"    Optimisation: {i}/{n} ({label})")

            opt_rows = _run_optimisation(
                cfg, quick=optimization_quick,
                trials=optimization_trials,
                progress_fn=_opt_progress)

            # Write CSV
            _write_optimisation_csv(opt_rows, csv_path)
            _log(f"  CSV saved: {os.path.abspath(csv_path)}")

            # Add results pages
            _optimisation_summary_page(pdf, opt_rows, top_n=15)
            _optimisation_heatmap_page(pdf, opt_rows)

        # ── Sensitivity analysis ────────────────────────────────────
        _log(f"  Running sensitivity analysis ({sensitivity_trials} "
             f"trials/point)...")

        sensitivity_params = [
            (("isr", "fov_deg"), "ISR FOV", "deg",
             [0.2, 0.5, 1.0, 2.0, 3.0, 5.0]),
            (("isr", "frame_rate_hz"), "ISR Frame Rate", "Hz",
             [5, 10, 20, 30, 50, 100]),
            (("cell", "radius_nm"), "Cell Radius", "nm",
             [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]),
            (("geometry", "slant_range_nm"), "Slant Range", "nm",
             [5.0, 8.0, 10.0, 14.0, 20.0]),
            (("platform", "altitude_ft"), "Altitude", "ft",
             [15000, 20000, 25000, 30000, 35000]),
            (("detection", "pd_in_fov"), "Base Pd", "",
             [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]),
            (("car", "speed_kts"), "Car Speed", "kts",
             [10, 20, 30, 50, 80]),
        ]

        # Title page for sensitivity section
        section_num = 10 if run_optimization else 8
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.75, f"{section_num}. Sensitivity Analysis",
                transform=ax.transAxes, fontsize=18, ha="center",
                fontweight="bold", va="top")
        sensitivity_intro = (
            "One-at-a-time (OAT) parameter sweeps\n\n"
            "Each chart below shows how detection rate and mean TTFD\n"
            "change when a single parameter is varied while all others\n"
            "remain at their default values.  This reveals which\n"
            "parameters have the greatest influence on ISR detection\n"
            "performance.\n\n"
            "Parameters swept:\n"
        )
        for _, label, unit, vals in sensitivity_params:
            lo, hi = min(vals), max(vals)
            sensitivity_intro += (f"  - {label}: {lo} - {hi} {unit} "
                                  f"({len(vals)} points)\n")
        sensitivity_intro += (f"\nMC trials per point: {sensitivity_trials}")
        ax.text(0.1, 0.60, sensitivity_intro, transform=ax.transAxes,
                fontsize=10, va="top", fontfamily="monospace",
                linespacing=1.4)
        pdf.savefig(fig)
        plt.close(fig)

        # CSV for sensitivity data
        sens_csv_rows = []

        for param_path, label, unit, values in sensitivity_params:
            _log(f"    Sweeping {label}...")

            def _sens_progress(i, n, info):
                _log(f"      [{i}/{n}] {info}")

            results = _run_sensitivity(
                cfg, param_path, label, values,
                mode="isr", trials=sensitivity_trials,
                progress_fn=_sens_progress)

            det_rates = [r["detection_rate"] for r in results]
            mean_ttfds = [r["mean_ttfd"] for r in results]
            vals = [r["value"] for r in results]

            _plot_sensitivity(pdf, label, unit, vals, det_rates,
                              mean_ttfds, mode_label="ISR")

            for r in results:
                sens_csv_rows.append({
                    "parameter": label,
                    "value": r["value"],
                    "unit": unit,
                    "detection_rate_pct": round(r["detection_rate"], 2),
                    "mean_ttfd_s": round(r["mean_ttfd"], 3) if np.isfinite(r["mean_ttfd"]) else "inf",
                })

        # Write sensitivity CSV
        if sens_csv_rows:
            with open(sensitivity_csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["parameter", "value", "unit",
                                    "detection_rate_pct", "mean_ttfd_s"])
                writer.writeheader()
                for row in sens_csv_rows:
                    writer.writerow(row)
            _log(f"  Sensitivity CSV saved: "
                 f"{os.path.abspath(sensitivity_csv_path)}")

        # ── Summary / key findings page ─────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.97, "Summary & Key Findings",
                transform=ax.transAxes, fontsize=16, ha="center",
                fontweight="bold", va="top")

        summary_lines = []

        # Sensitivity highlights
        summary_lines.append("Sensitivity Highlights")
        summary_lines.append("-" * 50)
        for param_path, label, unit, values in sensitivity_params:
            results = _run_sensitivity(
                cfg, param_path, label, values,
                mode="isr", trials=max(10, sensitivity_trials // 5))
            dets = [r["detection_rate"] for r in results]
            if dets:
                rng_det = max(dets) - min(dets)
                best_val = values[dets.index(max(dets))]
                summary_lines.append(
                    f"  {label:20s}: range {rng_det:5.1f}pp, "
                    f"best at {best_val} {unit}")
        summary_lines.append("")

        if opt_rows:
            valid = [r for r in opt_rows if not r.get("skipped", False)]
            if valid:
                valid.sort(key=lambda r: -r["detection_rate_pct"])
                best = valid[0]
                summary_lines.append("Best ISR Configuration Found (FOV fixed)")
                summary_lines.append("-" * 50)
                summary_lines.append(f"  Frame Rate: {best['frame_rate_hz']:.0f} Hz")
                summary_lines.append(f"  Cell R:     {best['cell_radius_nm']:.2f} nm")
                summary_lines.append(f"  Slant R:    {best['slant_range_nm']:.1f} nm")
                summary_lines.append(f"  Altitude:   {best['altitude_ft']:.0f} ft")
                summary_lines.append(f"  Detection:  {best['detection_rate_pct']:.1f}%")
                ttfd_str = (f"{best['mean_ttfd_s']:.1f}s"
                            if np.isfinite(best['mean_ttfd_s']) else "N/A")
                summary_lines.append(f"  Mean TTFD:  {ttfd_str}")
                summary_lines.append("")

        summary_lines.append("Output Files")
        summary_lines.append("-" * 50)
        summary_lines.append(f"  Report PDF:       {os.path.basename(pdf_path)}")
        if run_optimization:
            summary_lines.append(f"  Optimisation CSV: {os.path.basename(csv_path)}")
        summary_lines.append(f"  Sensitivity CSV:  {os.path.basename(sensitivity_csv_path)}")

        ax.text(0.05, 0.90, "\n".join(summary_lines),
                transform=ax.transAxes, fontsize=9, va="top",
                fontfamily="monospace", linespacing=1.35)

        elapsed = time.time() - t_start
        ax.text(0.5, 0.02, f"Report generated in {elapsed:.0f}s",
                transform=ax.transAxes, fontsize=9, ha="center",
                style="italic")
        pdf.savefig(fig)
        plt.close(fig)

    _log(f"Comprehensive report saved: {os.path.abspath(pdf_path)}")
    elapsed = time.time() - t_start
    _log(f"Total report generation time: {elapsed:.0f}s")

    output_files = {"pdf": pdf_path, "sensitivity_csv": sensitivity_csv_path}
    if run_optimization:
        output_files["optimization_csv"] = csv_path
    return output_files


# =====================================================================
# CLI entry point
# =====================================================================

def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Generate a comprehensive NTISR vs ISR report")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Output directory (default: ./results)")
    parser.add_argument("--no-optimization", action="store_true",
                        help="Skip the ISR optimisation sweep")
    parser.add_argument("--quick", action="store_true", default=True,
                        help="Use quick (coarse) optimisation grid (default)")
    parser.add_argument("--full", action="store_true",
                        help="Use full optimisation grid")
    parser.add_argument("--opt-trials", type=int, default=50,
                        help="MC trials per optimisation point (default: 50)")
    parser.add_argument("--sens-trials", type=int, default=50,
                        help="MC trials per sensitivity point (default: 50)")
    args = parser.parse_args()

    # Load config
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "config_default.yaml")
    config_path = args.config or default_path
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    quick = not args.full

    generate_comprehensive_report(
        cfg,
        output_dir=args.output_dir,
        run_optimization=not args.no_optimization,
        optimization_quick=quick,
        optimization_trials=args.opt_trials,
        sensitivity_trials=args.sens_trials,
    )


if __name__ == "__main__":
    main()
