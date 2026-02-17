#!/usr/bin/env python3
"""
NTISR vs ISR Operational Analysis — Monte Carlo Comparison Tool

Compares three airborne EO/IR sensor search modes against a moving ground
target (car) in a circular geographic cell:
  1. ISR with back-scanning mirror (large FOV, continuous sweep)
  2. NTISR step-and-stare (narrow FOV, raster scan)
  3. NTISR FMV search (narrow FOV, operator-guided panning)

Usage:
    python main.py                          # Run with defaults
    python main.py --altitude 25000         # Set platform altitude
    python main.py --config my_config.yaml  # Use custom config
    python main.py --help                   # Show all options
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np

from mc_engine import run_mc
from visualization import generate_pdf_report, generate_animation_frames
from interactive_menu import interactive_menu


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file, falling back to defaults."""
    default_path = os.path.join(os.path.dirname(__file__), 'config_default.yaml')
    with open(default_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_cfg = yaml.safe_load(f)
        if user_cfg:
            _deep_update(cfg, user_cfg)

    return cfg


def _deep_update(base: dict, override: dict):
    """Recursively update base dict with override values."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def apply_cli_overrides(cfg: dict, args: argparse.Namespace):
    """Apply CLI argument overrides to the config."""
    if args.altitude is not None:
        cfg['platform']['altitude_ft'] = args.altitude
    if args.speed is not None:
        cfg['platform']['speed_kts'] = args.speed
    if args.slant_range is not None:
        cfg['geometry']['slant_range_nm'] = args.slant_range
    if args.cell_radius is not None:
        cfg['cell']['radius_nm'] = args.cell_radius
    if args.car_speed is not None:
        cfg['car']['speed_kts'] = args.car_speed
    if args.isr_fov is not None:
        cfg['isr']['fov_deg'] = args.isr_fov
    if args.isr_frame_rate is not None:
        cfg['isr']['frame_rate_hz'] = args.isr_frame_rate
    if args.ntisr_ss_fov is not None:
        cfg['ntisr_step_stare']['fov_deg'] = args.ntisr_ss_fov
    if args.ntisr_ss_dwell is not None:
        cfg['ntisr_step_stare']['dwell_time_s'] = args.ntisr_ss_dwell
    if args.ntisr_ss_slew is not None:
        cfg['ntisr_step_stare']['slew_rate_deg_s'] = args.ntisr_ss_slew
    if args.ntisr_fmv_fov is not None:
        cfg['ntisr_fmv']['fov_deg'] = args.ntisr_fmv_fov
    if args.ntisr_fmv_slew is not None:
        cfg['ntisr_fmv']['slew_rate_deg_s'] = args.ntisr_fmv_slew
    if args.ntisr_fmv_road_bias is not None:
        cfg['ntisr_fmv']['road_bias'] = args.ntisr_fmv_road_bias
    if args.duration is not None:
        cfg['simulation']['duration_s'] = args.duration
    if args.dt is not None:
        cfg['simulation']['time_step_s'] = args.dt
    if args.trials is not None:
        cfg['simulation']['mc_trials'] = args.trials
    if args.seed is not None:
        cfg['simulation']['random_seed'] = args.seed
    if args.pd is not None:
        cfg['detection']['pd_in_fov'] = args.pd
    if args.no_animation:
        cfg['output']['animation'] = False
    if args.no_pdf:
        cfg['output']['pdf_report'] = False
    if args.output_dir is not None:
        cfg['output']['output_dir'] = args.output_dir


def print_config_summary(cfg: dict):
    """Print a formatted summary of the active configuration."""
    from geometry import ground_footprint_at_cell, NM_TO_FT
    alt = cfg['platform']['altitude_ft']
    sr = cfg['geometry']['slant_range_nm']

    print("=" * 65)
    print("  NTISR vs ISR Operational Analysis — Configuration Summary")
    print("=" * 65)
    print(f"  Platform:    {alt} ft AGL, {cfg['platform']['speed_kts']} kts")
    print(f"  Slant range: {sr} nm")
    print(f"  Cell radius: {cfg['cell']['radius_nm']} nm "
          f"({cfg['cell']['radius_nm'] * NM_TO_FT:.0f} ft)")
    print(f"  Car speed:   {cfg['car']['speed_kts']} kts")
    print()

    # Compute and display ground footprints
    sensors = [
        ("ISR FOV (1 frame)", cfg['isr']['fov_deg']),
        ("NTISR S&S FOV", cfg['ntisr_step_stare']['fov_deg']),
        ("NTISR FMV FOV", cfg['ntisr_fmv']['fov_deg']),
    ]
    print("  Ground footprints at cell center:")
    for name, fov in sensors:
        fp = ground_footprint_at_cell(alt, sr, fov)
        print(f"    {name:20s}: {fov:5.2f}° -> {fp:.4f} nm ({fp*NM_TO_FT:.0f} ft)")
    print()

    cell_area = np.pi * cfg['cell']['radius_nm']**2
    ss_fp = ground_footprint_at_cell(alt, sr, cfg['ntisr_step_stare']['fov_deg'])
    fmv_fp = ground_footprint_at_cell(alt, sr, cfg['ntisr_fmv']['fov_deg'])
    isr_fp = ground_footprint_at_cell(alt, sr, cfg['isr']['fov_deg'])

    # ISR raster: how many frames to cover cell, and scan-cycle time
    from sensors import ISRLineScan
    _tmp = ISRLineScan(alt, sr, 0, 0, cfg['cell']['radius_nm'], 0,
                       np.random.default_rng(0),
                       cfg['isr']['fov_deg'], cfg['isr']['frame_rate_hz'])
    n_positions = len(_tmp.scan_positions)
    cycle_s = _tmp.scan_cycle_time

    print(f"  Cell area:     {cell_area:.4f} nm² ({cell_area*NM_TO_FT**2:.0f} ft²)")
    print(f"  ISR coverage:  {(isr_fp**2/cell_area)*100:.2f}% per frame, "
          f"{n_positions} positions, {cycle_s:.2f}s/cycle")
    print(f"  S&S coverage:  {(ss_fp**2/cell_area)*100:.2f}% per stare")
    print(f"  FMV coverage:  {(fmv_fp**2/cell_area)*100:.2f}% per frame")
    print()
    print(f"  MC trials: {cfg['simulation']['mc_trials']}, "
          f"duration: {cfg['simulation']['duration_s']}s, "
          f"dt: {cfg['simulation']['time_step_s']}s")
    print(f"  Pd in FOV: {cfg['detection']['pd_in_fov']}, "
          f"min dwell: {cfg['detection']['min_dwell_for_detect_s']}s")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description='NTISR vs ISR Operational Analysis — Monte Carlo Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --altitude 25000 --trials 1000
  python main.py --car-speed 60 --cell-radius 0.5 --ntisr-ss-fov 0.3
  python main.py --config custom.yaml --no-animation
        """)

    # Platform parameters
    plat = parser.add_argument_group('Platform')
    plat.add_argument('--altitude', type=float, metavar='FT',
                      help='Platform altitude [ft] (default: 30000)')
    plat.add_argument('--speed', type=float, metavar='KTS',
                      help='Platform speed [kts] (default: 420)')
    plat.add_argument('--slant-range', type=float, metavar='NM',
                      help='Slant range to cell center [nm] (default: 10)')

    # Cell and target
    tgt = parser.add_argument_group('Cell & Target')
    tgt.add_argument('--cell-radius', type=float, metavar='NM',
                     help='Cell radius [nm] (default: 1.0)')
    tgt.add_argument('--car-speed', type=float, metavar='KTS',
                     help='Car speed [kts] (default: 30)')

    # ISR sensor
    isr = parser.add_argument_group('ISR Line Scanner (30 Hz raster)')
    isr.add_argument('--isr-fov', type=float, metavar='DEG',
                     help='ISR sensor FOV [deg] (default: 1.0)')
    isr.add_argument('--isr-frame-rate', type=float, metavar='HZ',
                     help='ISR frame rate [Hz] (default: 30)')

    # NTISR step-and-stare
    nss = parser.add_argument_group('NTISR Step-and-Stare')
    nss.add_argument('--ntisr-ss-fov', type=float, metavar='DEG',
                     help='NTISR S&S FOV [deg] (default: 0.5)')
    nss.add_argument('--ntisr-ss-dwell', type=float, metavar='SEC',
                     help='NTISR S&S dwell time [s] (default: 2.0)')
    nss.add_argument('--ntisr-ss-slew', type=float, metavar='DEG/S',
                     help='NTISR S&S gimbal slew rate [deg/s] (default: 20)')

    # NTISR FMV
    fmv = parser.add_argument_group('NTISR FMV Search')
    fmv.add_argument('--ntisr-fmv-fov', type=float, metavar='DEG',
                     help='NTISR FMV FOV [deg] (default: 1.0)')
    fmv.add_argument('--ntisr-fmv-slew', type=float, metavar='DEG/S',
                     help='NTISR FMV operator pan rate [deg/s] (default: 10)')
    fmv.add_argument('--ntisr-fmv-road-bias', type=float, metavar='0-1',
                     help='FMV operator road/feature bias (default: 0.7)')

    # Detection
    det = parser.add_argument_group('Detection Model')
    det.add_argument('--pd', type=float, metavar='0-1',
                     help='Base Pd when target is in FOV (default: 0.8)')

    # Simulation
    sim = parser.add_argument_group('Simulation')
    sim.add_argument('--duration', type=float, metavar='SEC',
                     help='Simulation duration [s] (default: 300)')
    sim.add_argument('--dt', type=float, metavar='SEC',
                     help='Time step [s] (default: 0.1)')
    sim.add_argument('--trials', type=int, metavar='N',
                     help='Number of MC trials (default: 500)')
    sim.add_argument('--seed', type=int,
                     help='Random seed (default: 42)')

    # Output
    out = parser.add_argument_group('Output')
    out.add_argument('--config', type=str, metavar='YAML',
                     help='Path to custom config YAML file')
    out.add_argument('--output-dir', type=str, metavar='DIR',
                     help='Output directory (default: ./results)')
    out.add_argument('--no-animation', action='store_true',
                     help='Skip animation frame generation')
    out.add_argument('--no-pdf', action='store_true',
                     help='Skip PDF report generation')
    out.add_argument('--interactive', '-i', action='store_true',
                     help='Launch interactive menu to configure and run')

    args = parser.parse_args()

    # Load and merge config
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)

    # ── Interactive mode ────────────────────────────────────────
    if args.interactive:
        result = interactive_menu(cfg)
        if result is None:
            print("Exiting.")
            sys.exit(0)
        cfg, selected_modes = result
    else:
        selected_modes = ['isr', 'ntisr_ss', 'ntisr_fmv']

    # Validate slant range > altitude
    alt_nm = cfg['platform']['altitude_ft'] / 6076.12
    if cfg['geometry']['slant_range_nm'] <= alt_nm:
        print(f"ERROR: Slant range ({cfg['geometry']['slant_range_nm']} nm) must be "
              f"greater than altitude ({alt_nm:.2f} nm / {cfg['platform']['altitude_ft']} ft)")
        sys.exit(1)

    print_config_summary(cfg)

    # Run Monte Carlo for each mode
    modes = selected_modes
    mode_names = {
        'isr': 'ISR (Back-Scan Mirror)',
        'ntisr_ss': 'NTISR (Step & Stare)',
        'ntisr_fmv': 'NTISR (FMV Search)'
    }

    results = {}
    for mode in modes:
        print(f"\nRunning MC for {mode_names[mode]}...")
        t0 = time.time()

        def progress(trial, total, _m=mode_names[mode]):
            if trial % max(1, total // 10) == 0 or trial == total:
                pct = trial / total * 100
                print(f"  {_m}: {trial}/{total} trials ({pct:.0f}%)")

        results[mode] = run_mc(cfg, mode, progress_callback=progress)
        elapsed = time.time() - t0
        r = results[mode]
        print(f"  Done in {elapsed:.1f}s — "
              f"detect rate: {r.overall_detect_fraction*100:.1f}%, "
              f"mean TTFD: {r.mean_ttfd:.1f}s")

    # Generate outputs
    output_dir = cfg['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    if cfg['output'].get('pdf_report', True):
        pdf_path = os.path.join(output_dir, 'ntisr_vs_isr_report.pdf')
        print(f"\nGenerating PDF report...")
        generate_pdf_report(results, cfg, pdf_path)

    if cfg['output'].get('animation', True):
        anim_dir = os.path.join(output_dir, 'animation_frames')
        print(f"\nGenerating animation frames...")
        generate_animation_frames(cfg, anim_dir)

    # Print final summary
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Mode':<25s} {'Detect %':>10s} {'Mean TTFD':>12s} "
          f"{'Median TTFD':>13s}")
    print("-" * 65)
    for mode in modes:
        r = results[mode]
        ttfd_mean = f"{r.mean_ttfd:.1f}s" if np.isfinite(r.mean_ttfd) else "Never"
        ttfd_med = f"{r.median_ttfd:.1f}s" if np.isfinite(r.median_ttfd) else "Never"
        print(f"  {mode_names[mode]:<25s} {r.overall_detect_fraction*100:>9.1f}% "
              f"{ttfd_mean:>12s} {ttfd_med:>13s}")
    print("=" * 65)
    print(f"\nOutputs saved to: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    main()
