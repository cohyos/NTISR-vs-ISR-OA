"""
Advanced analysis module for NTISR vs ISR Operational Analysis.

Provides:
  1. Duration crossover analysis — detection rate vs observation time
  2. Multi-platform comparison   — compare altitude/slant-range profiles
  3. Multi-target scenarios       — N targets in the cell simultaneously
  4. Moving platform orbit model  — racetrack orbit with time-varying geometry
  5. Per-trial raw data export    — CSV of every MC trial
  6. Parallel MC execution        — multiprocessing wrapper

All analysis functions accept a base config dict and return results
that can be plotted in the comprehensive report or exported to CSV.
"""

import copy
import csv
import os
import time
import multiprocessing
from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from geometry import FT_TO_NM, NM_TO_FT


# =====================================================================
# Parallel MC execution
# =====================================================================

def _trial_worker(args):
    """
    Execute a single MC trial.  Designed to be called via
    multiprocessing.Pool.map().

    Args:
        args: tuple of (trial_idx, mode, cfg, cell_cx, cell_cy, dt, n_steps, seed)

    Returns:
        dict with trial results: ttfd, total_detections, coverage_ratio,
        cumulative_pd array
    """
    from car_model import Car
    from mc_engine import build_sensor

    trial_idx, mode, cfg, cell_cx, cell_cy, dt, n_steps, base_seed = args
    n_targets = cfg.get('advanced', {}).get('n_targets', 1)

    trial_seed = (base_seed + trial_idx * 1000) if base_seed is not None else None
    rng = np.random.default_rng(trial_seed)

    cell_radius = cfg['cell']['radius_nm']
    cars = []
    for _ in range(n_targets):
        cars.append(Car(
            cell_center_x=cell_cx, cell_center_y=cell_cy,
            cell_radius_nm=cell_radius,
            speed_kts=cfg['car']['speed_kts'],
            heading_change_interval_s=cfg['car']['heading_change_interval_s'],
            rng=np.random.default_rng(
                rng.integers(0, 2**31) if trial_seed is not None else None)
        ))

    sensor = build_sensor(mode, cfg, cell_cx, cell_cy, rng)

    # Per-target tracking
    detected_any = False
    ttfd = np.inf
    total_detections = 0
    per_target_detected = [False] * n_targets
    per_target_ttfd = [np.inf] * n_targets
    cumulative_pd = np.zeros(n_steps)

    for step_i in range(n_steps):
        t = step_i * dt
        for car in cars:
            car.step(dt)
        sensor.step(t, dt)

        for ci, car in enumerate(cars):
            cx, cy = car.get_position()
            if sensor.check_detection(cx, cy, dt):
                total_detections += 1
                if not per_target_detected[ci]:
                    per_target_detected[ci] = True
                    per_target_ttfd[ci] = t
                if not detected_any:
                    detected_any = True
                    ttfd = t

        if detected_any:
            cumulative_pd[step_i] = 1.0

    return {
        'trial_idx': trial_idx,
        'ttfd': ttfd,
        'total_detections': total_detections,
        'coverage_ratio': sensor.coverage_ratio,
        'cumulative_pd': cumulative_pd,
        'per_target_ttfd': per_target_ttfd,
        'n_targets_detected': sum(per_target_detected),
    }


def run_mc_parallel(cfg, mode, n_workers=None, progress_fn=None):
    """
    Run Monte Carlo simulation using multiprocessing.

    Falls back to single-threaded if n_workers=1 or if the pool
    fails to start.

    Args:
        cfg: configuration dict
        mode: sensor mode key
        n_workers: number of processes (None = cpu_count)
        progress_fn: optional callable(completed, total)

    Returns:
        MCResult with all statistics populated
    """
    from mc_engine import MCResult

    n_trials = cfg['simulation']['mc_trials']
    duration = cfg['simulation']['duration_s']
    dt = cfg['simulation']['time_step_s']
    n_steps = int(duration / dt)
    seed = cfg['simulation'].get('random_seed', None)

    cell_cx, cell_cy = 0.0, 0.0

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    if n_workers <= 1 or n_trials <= 4:
        # Fall back to single-threaded for small jobs
        from mc_engine import run_mc
        return run_mc(cfg, mode, progress_callback=progress_fn)

    # Build argument list
    args_list = [
        (i, mode, cfg, cell_cx, cell_cy, dt, n_steps, seed)
        for i in range(n_trials)
    ]

    result = MCResult(mode, n_trials, n_steps, dt)
    coverage_ratios = []

    try:
        with multiprocessing.Pool(n_workers) as pool:
            for i, trial_result in enumerate(
                    pool.imap_unordered(_trial_worker, args_list)):
                idx = trial_result['trial_idx']
                result.time_to_first_detect[idx] = trial_result['ttfd']
                result.total_detections[idx] = trial_result['total_detections']
                result.cumulative_pd[idx] = trial_result['cumulative_pd']
                coverage_ratios.append(trial_result['coverage_ratio'])

                if progress_fn and ((i + 1) % max(1, n_trials // 20) == 0
                                    or i + 1 == n_trials):
                    progress_fn(i + 1, n_trials)
    except Exception:
        # Fallback to sequential
        from mc_engine import run_mc
        return run_mc(cfg, mode, progress_callback=progress_fn)

    avg_cov = np.mean(coverage_ratios) if coverage_ratios else 0.0
    result.finalize(coverage_ratio=avg_cov)
    return result


# =====================================================================
# Per-trial raw data export
# =====================================================================

def export_per_trial_csv(results, output_path):
    """
    Export per-trial raw data for all sensor modes to a CSV file.

    Args:
        results: dict mapping mode_key -> MCResult
        output_path: path to output CSV
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    mode_names = {
        'isr': 'ISR', 'ntisr_ss': 'NTISR_SS', 'ntisr_fmv': 'NTISR_FMV',
    }
    columns = ['mode', 'trial', 'detected', 'ttfd_s', 'total_detections']

    with open(output_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for mode_key, mc in results.items():
            label = mode_names.get(mode_key, mode_key)
            for i in range(mc.n_trials):
                ttfd = mc.time_to_first_detect[i]
                writer.writerow({
                    'mode': label,
                    'trial': i,
                    'detected': 1 if np.isfinite(ttfd) else 0,
                    'ttfd_s': round(ttfd, 3) if np.isfinite(ttfd) else '',
                    'total_detections': int(mc.total_detections[i]),
                })
    print(f"  Per-trial CSV saved: {os.path.abspath(output_path)}")


# =====================================================================
# Duration crossover analysis
# =====================================================================

def run_duration_crossover(cfg, modes=None, durations=None,
                           trials_per_point=50, n_workers=None,
                           progress_fn=None):
    """
    Run MC at multiple observation durations to find the crossover point
    where slower sensors catch up to faster ones.

    Args:
        cfg: base configuration dict
        modes: list of mode keys (default: all three)
        durations: list of duration values in seconds
        trials_per_point: MC trials per (mode, duration) pair
        n_workers: parallel workers (None = auto)
        progress_fn: optional callable(message)

    Returns:
        dict with keys:
          'durations': list of durations
          'results': dict[mode][duration] -> {detection_rate, mean_ttfd, dti, sei}
    """
    if modes is None:
        modes = ['isr', 'ntisr_ss', 'ntisr_fmv']
    if durations is None:
        durations = [30, 60, 120, 180, 300, 450, 600]

    all_results = {m: {} for m in modes}
    total = len(modes) * len(durations)
    done = 0

    for dur in durations:
        for mode in modes:
            done += 1
            if progress_fn:
                progress_fn(f"  Duration crossover: {mode} @ {dur}s "
                            f"[{done}/{total}]")

            test_cfg = copy.deepcopy(cfg)
            test_cfg['simulation']['duration_s'] = dur
            test_cfg['simulation']['mc_trials'] = trials_per_point

            mc = run_mc_parallel(test_cfg, mode, n_workers=n_workers)
            all_results[mode][dur] = {
                'detection_rate': mc.overall_detect_fraction * 100,
                'mean_ttfd': mc.mean_ttfd,
                'dti': mc.dti,
                'sei': mc.sei,
            }

    return {'durations': durations, 'results': all_results}


def plot_duration_crossover(pdf, crossover_data, modes=None):
    """Add duration crossover charts to a PDF."""
    durations = crossover_data['durations']
    results = crossover_data['results']

    mode_names = {
        'isr': 'ISR (Back-Scan)',
        'ntisr_ss': 'NTISR (Step & Stare)',
        'ntisr_fmv': 'NTISR (FMV)',
    }
    colors = {'isr': '#4472C4', 'ntisr_ss': '#ED7D31', 'ntisr_fmv': '#70AD47'}

    if modes is None:
        modes = list(results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Panel 1: Detection rate vs duration
    for m in modes:
        rates = [results[m][d]['detection_rate'] for d in durations]
        ax1.plot(durations, rates, 'o-', color=colors.get(m, 'gray'),
                 linewidth=2, markersize=5, label=mode_names.get(m, m))
    ax1.set_xlabel('Observation Duration [s]', fontsize=11)
    ax1.set_ylabel('Detection Rate [%]', fontsize=11)
    ax1.set_title('Detection Rate vs Duration', fontsize=12,
                  fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # Panel 2: Mean TTFD vs duration
    for m in modes:
        ttfds = []
        valid_durs = []
        for d in durations:
            t = results[m][d]['mean_ttfd']
            if np.isfinite(t):
                ttfds.append(t)
                valid_durs.append(d)
        if valid_durs:
            ax2.plot(valid_durs, ttfds, 's-', color=colors.get(m, 'gray'),
                     linewidth=2, markersize=5, label=mode_names.get(m, m))
    ax2.set_xlabel('Observation Duration [s]', fontsize=11)
    ax2.set_ylabel('Mean TTFD [s]', fontsize=11)
    ax2.set_title('Mean TTFD vs Duration', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    fig.suptitle('Duration Crossover Analysis', fontsize=14,
                 fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2: DTI and SEI vs duration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for m in modes:
        dtis = [results[m][d]['dti'] for d in durations]
        ax1.plot(durations, dtis, 'o-', color=colors.get(m, 'gray'),
                 linewidth=2, markersize=5, label=mode_names.get(m, m))
    ax1.set_xlabel('Observation Duration [s]', fontsize=11)
    ax1.set_ylabel('DTI', fontsize=11)
    ax1.set_title('Detection Timeliness Index vs Duration', fontsize=12,
                  fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    for m in modes:
        seis = [results[m][d]['sei'] for d in durations]
        ax2.plot(durations, seis, 's-', color=colors.get(m, 'gray'),
                 linewidth=2, markersize=5, label=mode_names.get(m, m))
    ax2.set_xlabel('Observation Duration [s]', fontsize=11)
    ax2.set_ylabel('SEI = DTI / Coverage', fontsize=11)
    ax2.set_title('Search Efficiency Index vs Duration', fontsize=12,
                  fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    fig.suptitle('Efficiency Metrics vs Observation Duration', fontsize=14,
                 fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# =====================================================================
# Multi-platform comparison
# =====================================================================

PLATFORM_PROFILES = {
    'high_alt_isr': {
        'label': 'High-Alt ISR (45kft / 20nm)',
        'altitude_ft': 45000,
        'slant_range_nm': 20.0,
    },
    'mid_alt_isr': {
        'label': 'Mid-Alt ISR (30kft / 10nm)',
        'altitude_ft': 30000,
        'slant_range_nm': 10.0,
    },
    'low_alt_ntisr': {
        'label': 'Low-Alt NTISR (15kft / 5nm)',
        'altitude_ft': 15000,
        'slant_range_nm': 5.0,
    },
    'fighter_pod': {
        'label': 'Fighter Pod (20kft / 8nm)',
        'altitude_ft': 20000,
        'slant_range_nm': 8.0,
    },
}


def run_multi_platform(cfg, profiles=None, modes=None,
                       trials_per_point=50, n_workers=None,
                       progress_fn=None):
    """
    Run MC for each (platform_profile, sensor_mode) combination.

    Args:
        cfg: base config dict
        profiles: dict of profile_key -> {altitude_ft, slant_range_nm, label}
                  (defaults to PLATFORM_PROFILES)
        modes: sensor modes to compare
        trials_per_point: MC trials per combination
        n_workers: parallel workers
        progress_fn: optional callable(message)

    Returns:
        dict with keys:
          'profiles': ordered list of profile keys
          'profile_labels': dict[key] -> label
          'modes': list of mode keys
          'results': dict[profile_key][mode] -> {detection_rate, mean_ttfd, dti, sei}
    """
    if profiles is None:
        profiles = PLATFORM_PROFILES
    if modes is None:
        modes = ['isr', 'ntisr_ss', 'ntisr_fmv']

    profile_keys = list(profiles.keys())
    all_results = {pk: {} for pk in profile_keys}
    total = len(profile_keys) * len(modes)
    done = 0

    for pk in profile_keys:
        prof = profiles[pk]
        alt = prof['altitude_ft']
        sr = prof['slant_range_nm']

        # Validate geometry
        alt_nm = alt * FT_TO_NM
        if sr <= alt_nm:
            for mode in modes:
                all_results[pk][mode] = {
                    'detection_rate': 0.0, 'mean_ttfd': np.inf,
                    'dti': 0.0, 'sei': 0.0,
                }
                done += 1
            continue

        for mode in modes:
            done += 1
            if progress_fn:
                progress_fn(f"  Multi-platform: {prof['label']} / {mode} "
                            f"[{done}/{total}]")

            test_cfg = copy.deepcopy(cfg)
            test_cfg['platform']['altitude_ft'] = alt
            test_cfg['geometry']['slant_range_nm'] = sr
            test_cfg['simulation']['mc_trials'] = trials_per_point

            try:
                mc = run_mc_parallel(test_cfg, mode, n_workers=n_workers)
                all_results[pk][mode] = {
                    'detection_rate': mc.overall_detect_fraction * 100,
                    'mean_ttfd': mc.mean_ttfd,
                    'dti': mc.dti,
                    'sei': mc.sei,
                }
            except Exception:
                all_results[pk][mode] = {
                    'detection_rate': 0.0, 'mean_ttfd': np.inf,
                    'dti': 0.0, 'sei': 0.0,
                }

    return {
        'profiles': profile_keys,
        'profile_labels': {k: v['label'] for k, v in profiles.items()},
        'modes': modes,
        'results': all_results,
    }


def plot_multi_platform(pdf, mp_data):
    """Add multi-platform comparison charts to a PDF."""
    profiles = mp_data['profiles']
    labels = mp_data['profile_labels']
    modes = mp_data['modes']
    results = mp_data['results']

    mode_names = {
        'isr': 'ISR', 'ntisr_ss': 'NTISR S&S', 'ntisr_fmv': 'NTISR FMV',
    }
    colors = {'isr': '#4472C4', 'ntisr_ss': '#ED7D31', 'ntisr_fmv': '#70AD47'}

    n_profiles = len(profiles)
    n_modes = len(modes)
    x = np.arange(n_profiles)
    bar_w = 0.8 / n_modes

    # Page 1: Detection rate grouped bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))

    for i, m in enumerate(modes):
        rates = [results[pk][m]['detection_rate'] for pk in profiles]
        offset = (i - (n_modes - 1) / 2) * bar_w
        ax1.bar(x + offset, rates, bar_w * 0.9,
                color=colors.get(m, 'gray'), alpha=0.85,
                label=mode_names.get(m, m))
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels[pk] for pk in profiles],
                        rotation=20, ha='right', fontsize=8)
    ax1.set_ylabel('Detection Rate [%]', fontsize=11)
    ax1.set_title('Detection Rate by Platform', fontsize=12,
                  fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 105)

    # Panel 2: SEI
    for i, m in enumerate(modes):
        seis = [results[pk][m]['sei'] for pk in profiles]
        offset = (i - (n_modes - 1) / 2) * bar_w
        ax2.bar(x + offset, seis, bar_w * 0.9,
                color=colors.get(m, 'gray'), alpha=0.85,
                label=mode_names.get(m, m))
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[pk] for pk in profiles],
                        rotation=20, ha='right', fontsize=8)
    ax2.set_ylabel('SEI = DTI / Coverage', fontsize=11)
    ax2.set_title('Search Efficiency by Platform', fontsize=12,
                  fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)

    fig.suptitle('Multi-Platform Comparison', fontsize=14,
                 fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2: Summary table
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.97, 'Multi-Platform Comparison — Summary Table',
            transform=ax.transAxes, fontsize=14, ha='center',
            fontweight='bold', va='top')

    col_labels = ['Platform'] + [mode_names.get(m, m) + '\nDet%'
                                 for m in modes] + \
                 [mode_names.get(m, m) + '\nTTFD'
                  for m in modes]

    rows = []
    for pk in profiles:
        row = [labels[pk]]
        for m in modes:
            row.append(f"{results[pk][m]['detection_rate']:.1f}%")
        for m in modes:
            t = results[pk][m]['mean_ttfd']
            row.append(f"{t:.1f}s" if np.isfinite(t) else "N/A")
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    for (ri, ci), cell in table.get_celld().items():
        if ri == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif ri % 2 == 0:
            cell.set_facecolor('#D6E4F0')

    pdf.savefig(fig)
    plt.close(fig)


# =====================================================================
# Multi-target analysis
# =====================================================================

def run_multi_target(cfg, target_counts=None, modes=None,
                     trials_per_point=50, n_workers=None,
                     progress_fn=None):
    """
    Run MC with varying numbers of targets in the cell.

    For each target count, tracks:
      - Detection rate (detect ANY target)
      - Mean fraction of targets detected

    Args:
        cfg: base config dict
        target_counts: list of N values (default: [1, 2, 3, 5, 8])
        modes: sensor modes
        trials_per_point: MC trials per point
        n_workers: parallel workers
        progress_fn: optional callable(message)

    Returns:
        dict with 'target_counts', 'modes',
        'results': dict[mode][n] -> {detect_any_rate, mean_frac_detected, mean_ttfd}
    """
    if target_counts is None:
        target_counts = [1, 2, 3, 5, 8]
    if modes is None:
        modes = ['isr', 'ntisr_ss', 'ntisr_fmv']

    all_results = {m: {} for m in modes}
    total = len(modes) * len(target_counts)
    done = 0

    for n_tgt in target_counts:
        for mode in modes:
            done += 1
            if progress_fn:
                progress_fn(f"  Multi-target: {mode} / {n_tgt} targets "
                            f"[{done}/{total}]")

            test_cfg = copy.deepcopy(cfg)
            test_cfg['simulation']['mc_trials'] = trials_per_point
            test_cfg.setdefault('advanced', {})['n_targets'] = n_tgt

            # Run trials via the parallel worker which supports multi-target
            duration = test_cfg['simulation']['duration_s']
            dt = test_cfg['simulation']['time_step_s']
            n_steps = int(duration / dt)
            seed = test_cfg['simulation'].get('random_seed', None)

            args_list = [
                (i, mode, test_cfg, 0.0, 0.0, dt, n_steps, seed)
                for i in range(trials_per_point)
            ]

            detect_any = 0
            frac_detected = []
            ttfds = []

            try:
                if n_workers and n_workers > 1 and trials_per_point > 4:
                    with multiprocessing.Pool(
                            min(n_workers, multiprocessing.cpu_count())) as pool:
                        trial_results = pool.map(_trial_worker, args_list)
                else:
                    trial_results = [_trial_worker(a) for a in args_list]

                for tr in trial_results:
                    if np.isfinite(tr['ttfd']):
                        detect_any += 1
                        ttfds.append(tr['ttfd'])
                    frac_detected.append(
                        tr['n_targets_detected'] / n_tgt)
            except Exception:
                trial_results = [_trial_worker(a) for a in args_list]
                for tr in trial_results:
                    if np.isfinite(tr['ttfd']):
                        detect_any += 1
                        ttfds.append(tr['ttfd'])
                    frac_detected.append(
                        tr['n_targets_detected'] / n_tgt)

            all_results[mode][n_tgt] = {
                'detect_any_rate': detect_any / trials_per_point * 100,
                'mean_frac_detected': np.mean(frac_detected) * 100,
                'mean_ttfd': np.mean(ttfds) if ttfds else np.inf,
            }

    return {
        'target_counts': target_counts,
        'modes': modes,
        'results': all_results,
    }


def plot_multi_target(pdf, mt_data):
    """Add multi-target analysis charts to a PDF."""
    counts = mt_data['target_counts']
    modes = mt_data['modes']
    results = mt_data['results']

    mode_names = {
        'isr': 'ISR (Back-Scan)',
        'ntisr_ss': 'NTISR (Step & Stare)',
        'ntisr_fmv': 'NTISR (FMV)',
    }
    colors = {'isr': '#4472C4', 'ntisr_ss': '#ED7D31', 'ntisr_fmv': '#70AD47'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Panel 1: Detect-any rate vs target count
    for m in modes:
        rates = [results[m][n]['detect_any_rate'] for n in counts]
        ax1.plot(counts, rates, 'o-', color=colors.get(m, 'gray'),
                 linewidth=2, markersize=6, label=mode_names.get(m, m))
    ax1.set_xlabel('Number of Targets', fontsize=11)
    ax1.set_ylabel('Detect-Any Rate [%]', fontsize=11)
    ax1.set_title('P(detect at least one target)', fontsize=12,
                  fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    ax1.set_xticks(counts)

    # Panel 2: Mean fraction of targets detected
    for m in modes:
        fracs = [results[m][n]['mean_frac_detected'] for n in counts]
        ax2.plot(counts, fracs, 's-', color=colors.get(m, 'gray'),
                 linewidth=2, markersize=6, label=mode_names.get(m, m))
    ax2.set_xlabel('Number of Targets', fontsize=11)
    ax2.set_ylabel('Mean % of Targets Detected', fontsize=11)
    ax2.set_title('Average Fraction of Targets Found', fontsize=12,
                  fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    ax2.set_xticks(counts)

    fig.suptitle('Multi-Target Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# =====================================================================
# Moving platform (racetrack orbit) analysis
# =====================================================================

def _racetrack_slant_range(t, altitude_ft, base_slant_nm,
                           orbit_radius_nm, orbit_period_s, cell_offset_nm):
    """
    Compute the instantaneous slant range from a platform on a
    racetrack orbit to a fixed cell center.

    The platform flies a circular orbit of radius orbit_radius_nm
    centered at an offset from the cell.  The slant range varies
    as the platform moves around the orbit.

    Returns: slant_range_nm at time t
    """
    alt_nm = altitude_ft * FT_TO_NM
    # Platform position on circular orbit (simplified racetrack)
    angle = 2 * np.pi * t / orbit_period_s
    px = orbit_radius_nm * np.cos(angle) + cell_offset_nm
    py = orbit_radius_nm * np.sin(angle)
    ground_range = np.sqrt(px**2 + py**2)
    return np.sqrt(alt_nm**2 + ground_range**2)


def run_moving_platform(cfg, orbit_params=None, modes=None,
                        trials_per_point=50, progress_fn=None):
    """
    Compare detection performance between stationary and orbiting platforms.

    For the moving platform, the slant range varies over time as the
    platform flies a racetrack orbit.  We approximate this by sampling
    the orbit at several phases and averaging the results.

    Args:
        cfg: base config
        orbit_params: dict with orbit_radius_nm, orbit_period_s, cell_offset_nm
        modes: sensor modes
        trials_per_point: MC trials
        progress_fn: optional callable(message)

    Returns:
        dict with 'modes', 'stationary', 'orbit_samples', 'orbit_average'
    """
    if modes is None:
        modes = ['isr', 'ntisr_ss', 'ntisr_fmv']
    if orbit_params is None:
        orbit_params = {
            'orbit_radius_nm': 5.0,
            'orbit_period_s': 300.0,
            'cell_offset_nm': 8.0,
        }

    alt = cfg['platform']['altitude_ft']
    alt_nm = alt * FT_TO_NM

    # Sample orbit at N phases
    n_phases = 6
    phase_times = np.linspace(0, orbit_params['orbit_period_s'],
                              n_phases, endpoint=False)
    phase_slant_ranges = []
    for t in phase_times:
        sr = _racetrack_slant_range(
            t, alt,
            cfg['geometry']['slant_range_nm'],
            orbit_params['orbit_radius_nm'],
            orbit_params['orbit_period_s'],
            orbit_params['cell_offset_nm'],
        )
        phase_slant_ranges.append(sr)

    # Run stationary baseline
    stationary = {}
    for mode in modes:
        if progress_fn:
            progress_fn(f"  Orbit analysis: stationary / {mode}")
        test_cfg = copy.deepcopy(cfg)
        test_cfg['simulation']['mc_trials'] = trials_per_point
        mc = run_mc_parallel(test_cfg, mode, n_workers=1)
        stationary[mode] = {
            'detection_rate': mc.overall_detect_fraction * 100,
            'mean_ttfd': mc.mean_ttfd,
            'dti': mc.dti,
        }

    # Run at each orbit phase
    orbit_samples = {m: [] for m in modes}
    for i, (pt, sr) in enumerate(zip(phase_times, phase_slant_ranges)):
        if sr <= alt_nm:
            for m in modes:
                orbit_samples[m].append({
                    'phase_time': pt, 'slant_range': sr,
                    'detection_rate': 0, 'mean_ttfd': np.inf, 'dti': 0,
                })
            continue

        for mode in modes:
            if progress_fn:
                progress_fn(f"  Orbit analysis: phase {i+1}/{n_phases} / "
                            f"{mode} (SR={sr:.1f}nm)")
            test_cfg = copy.deepcopy(cfg)
            test_cfg['geometry']['slant_range_nm'] = sr
            test_cfg['simulation']['mc_trials'] = trials_per_point
            try:
                mc = run_mc_parallel(test_cfg, mode, n_workers=1)
                orbit_samples[mode].append({
                    'phase_time': pt, 'slant_range': sr,
                    'detection_rate': mc.overall_detect_fraction * 100,
                    'mean_ttfd': mc.mean_ttfd,
                    'dti': mc.dti,
                })
            except Exception:
                orbit_samples[mode].append({
                    'phase_time': pt, 'slant_range': sr,
                    'detection_rate': 0, 'mean_ttfd': np.inf, 'dti': 0,
                })

    # Compute orbit average
    orbit_average = {}
    for m in modes:
        dets = [s['detection_rate'] for s in orbit_samples[m]]
        dtis = [s['dti'] for s in orbit_samples[m]]
        ttfds = [s['mean_ttfd'] for s in orbit_samples[m]
                 if np.isfinite(s['mean_ttfd'])]
        orbit_average[m] = {
            'detection_rate': np.mean(dets),
            'mean_ttfd': np.mean(ttfds) if ttfds else np.inf,
            'dti': np.mean(dtis),
        }

    return {
        'modes': modes,
        'orbit_params': orbit_params,
        'phase_slant_ranges': phase_slant_ranges,
        'stationary': stationary,
        'orbit_samples': orbit_samples,
        'orbit_average': orbit_average,
    }


def plot_moving_platform(pdf, orbit_data):
    """Add moving platform analysis charts to a PDF."""
    modes = orbit_data['modes']
    stationary = orbit_data['stationary']
    orbit_avg = orbit_data['orbit_average']
    orbit_samples = orbit_data['orbit_samples']
    params = orbit_data['orbit_params']

    mode_names = {
        'isr': 'ISR', 'ntisr_ss': 'NTISR S&S', 'ntisr_fmv': 'NTISR FMV',
    }
    colors = {'isr': '#4472C4', 'ntisr_ss': '#ED7D31', 'ntisr_fmv': '#70AD47'}

    n_modes = len(modes)
    x = np.arange(n_modes)
    bar_w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))

    # Detection rate: stationary vs orbit average
    stat_rates = [stationary[m]['detection_rate'] for m in modes]
    orb_rates = [orbit_avg[m]['detection_rate'] for m in modes]

    ax1.bar(x - bar_w / 2, stat_rates, bar_w, color='#4472C4', alpha=0.85,
            label='Stationary')
    ax1.bar(x + bar_w / 2, orb_rates, bar_w, color='#ED7D31', alpha=0.85,
            label='Orbit Average')
    ax1.set_xticks(x)
    ax1.set_xticklabels([mode_names.get(m, m) for m in modes], fontsize=9)
    ax1.set_ylabel('Detection Rate [%]', fontsize=11)
    ax1.set_title('Stationary vs Orbiting Platform', fontsize=12,
                  fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 105)

    # Detection rate vs slant range across orbit phases
    srs = orbit_data['phase_slant_ranges']
    for m in modes:
        phase_dets = [s['detection_rate'] for s in orbit_samples[m]]
        ax2.plot(srs, phase_dets, 'o-', color=colors.get(m, 'gray'),
                 linewidth=2, markersize=5, label=mode_names.get(m, m))
    ax2.set_xlabel('Slant Range [nm]', fontsize=11)
    ax2.set_ylabel('Detection Rate [%]', fontsize=11)
    ax2.set_title('Detection Rate Along Orbit', fontsize=12,
                  fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    orbit_desc = (f"Orbit: R={params['orbit_radius_nm']}nm, "
                  f"T={params['orbit_period_s']}s, "
                  f"offset={params['cell_offset_nm']}nm")
    fig.suptitle(f'Moving Platform Analysis\n{orbit_desc}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# =====================================================================
# Combined CSV export for all analyses
# =====================================================================

def export_crossover_csv(crossover_data, path):
    """Export duration crossover results to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    durations = crossover_data['durations']
    results = crossover_data['results']
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=[
            'duration_s', 'mode', 'detection_rate_pct', 'mean_ttfd_s',
            'dti', 'sei'])
        w.writeheader()
        for d in durations:
            for m, data in results.items():
                r = data[d]
                w.writerow({
                    'duration_s': d, 'mode': m,
                    'detection_rate_pct': round(r['detection_rate'], 2),
                    'mean_ttfd_s': round(r['mean_ttfd'], 3)
                    if np.isfinite(r['mean_ttfd']) else 'inf',
                    'dti': round(r['dti'], 4),
                    'sei': round(r['sei'], 6),
                })


def export_multi_platform_csv(mp_data, path):
    """Export multi-platform comparison to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=[
            'platform', 'mode', 'detection_rate_pct', 'mean_ttfd_s',
            'dti', 'sei'])
        w.writeheader()
        for pk in mp_data['profiles']:
            for m in mp_data['modes']:
                r = mp_data['results'][pk][m]
                w.writerow({
                    'platform': mp_data['profile_labels'][pk],
                    'mode': m,
                    'detection_rate_pct': round(r['detection_rate'], 2),
                    'mean_ttfd_s': round(r['mean_ttfd'], 3)
                    if np.isfinite(r['mean_ttfd']) else 'inf',
                    'dti': round(r['dti'], 4),
                    'sei': round(r['sei'], 6),
                })


def export_multi_target_csv(mt_data, path):
    """Export multi-target analysis to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=[
            'n_targets', 'mode', 'detect_any_rate_pct',
            'mean_frac_detected_pct', 'mean_ttfd_s'])
        w.writeheader()
        for n in mt_data['target_counts']:
            for m in mt_data['modes']:
                r = mt_data['results'][m][n]
                w.writerow({
                    'n_targets': n, 'mode': m,
                    'detect_any_rate_pct': round(r['detect_any_rate'], 2),
                    'mean_frac_detected_pct': round(
                        r['mean_frac_detected'], 2),
                    'mean_ttfd_s': round(r['mean_ttfd'], 3)
                    if np.isfinite(r['mean_ttfd']) else 'inf',
                })


# =====================================================================
# Master analysis runner
# =====================================================================

def run_all_advanced_analyses(cfg, output_dir='./results',
                              trials_per_point=50,
                              n_workers=None,
                              progress_fn=None):
    """
    Run all advanced analyses and generate a comprehensive PDF + CSVs.

    Args:
        cfg: base configuration dict
        output_dir: output directory
        trials_per_point: MC trials per data point
        n_workers: parallel workers
        progress_fn: optional callable(message)

    Returns:
        dict with paths to generated files
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, 'advanced_analysis.pdf')

    def _log(msg):
        if progress_fn:
            progress_fn(msg)
        print(msg)

    _log("Starting advanced analyses...")
    t_start = time.time()

    modes = ['isr', 'ntisr_ss', 'ntisr_fmv']

    # 1. Duration crossover
    _log("\n[1/4] Duration Crossover Analysis")
    crossover = run_duration_crossover(
        cfg, modes=modes, trials_per_point=trials_per_point,
        n_workers=n_workers, progress_fn=_log)
    csv_cross = os.path.join(output_dir, 'duration_crossover.csv')
    export_crossover_csv(crossover, csv_cross)
    _log(f"  CSV: {csv_cross}")

    # 2. Multi-platform comparison
    _log("\n[2/4] Multi-Platform Comparison")
    mp_data = run_multi_platform(
        cfg, modes=modes, trials_per_point=trials_per_point,
        n_workers=n_workers, progress_fn=_log)
    csv_mp = os.path.join(output_dir, 'multi_platform.csv')
    export_multi_platform_csv(mp_data, csv_mp)
    _log(f"  CSV: {csv_mp}")

    # 3. Multi-target analysis
    _log("\n[3/4] Multi-Target Analysis")
    mt_data = run_multi_target(
        cfg, modes=modes, trials_per_point=trials_per_point,
        n_workers=n_workers, progress_fn=_log)
    csv_mt = os.path.join(output_dir, 'multi_target.csv')
    export_multi_target_csv(mt_data, csv_mt)
    _log(f"  CSV: {csv_mt}")

    # 4. Moving platform analysis
    _log("\n[4/4] Moving Platform (Orbit) Analysis")
    orbit_data = run_moving_platform(
        cfg, modes=modes, trials_per_point=trials_per_point,
        progress_fn=_log)

    # Generate PDF
    _log("\nGenerating advanced analysis PDF...")
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.5, 0.80, "Advanced Analysis Report",
                transform=ax.transAxes, fontsize=20, ha='center',
                fontweight='bold', va='top')
        ax.text(0.5, 0.70,
                "Duration Crossover | Multi-Platform | Multi-Target | "
                "Moving Platform",
                transform=ax.transAxes, fontsize=11, ha='center',
                style='italic')
        import datetime
        ax.text(0.5, 0.62,
                f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
                transform=ax.transAxes, fontsize=10, ha='center')

        config_lines = [
            f"Base config: {cfg['platform']['altitude_ft']:.0f} ft, "
            f"SR={cfg['geometry']['slant_range_nm']} nm, "
            f"Cell R={cfg['cell']['radius_nm']} nm",
            f"MC trials per point: {trials_per_point}",
            f"Duration: {cfg['simulation']['duration_s']}s, "
            f"dt={cfg['simulation']['time_step_s']}s",
        ]
        ax.text(0.1, 0.50, '\n'.join(config_lines),
                transform=ax.transAxes, fontsize=9, va='top',
                fontfamily='monospace')
        pdf.savefig(fig)
        plt.close(fig)

        # Analysis pages
        plot_duration_crossover(pdf, crossover, modes)
        plot_multi_platform(pdf, mp_data)
        plot_multi_target(pdf, mt_data)
        plot_moving_platform(pdf, orbit_data)

    elapsed = time.time() - t_start
    _log(f"\nAdvanced analysis complete in {elapsed:.0f}s")
    _log(f"  PDF:  {os.path.abspath(pdf_path)}")

    return {
        'pdf': pdf_path,
        'crossover_csv': csv_cross,
        'multi_platform_csv': csv_mp,
        'multi_target_csv': csv_mt,
    }
