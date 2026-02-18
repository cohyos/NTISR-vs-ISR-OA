"""
Interactive terminal menu for NTISR vs ISR Operational Analysis.

Provides a numbered-menu interface so users can view/edit parameters,
select which sensor modes to run, and launch the simulation without
memorising CLI flags.
"""

import os
import sys
import copy
import yaml


# ── helpers ──────────────────────────────────────────────────────────

def _clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def _pause(msg="Press Enter to continue..."):
    input(msg)


def _read_float(prompt, current, lo=None, hi=None):
    """Prompt for a float, returning *current* on empty input."""
    while True:
        raw = input(f"  {prompt} [{current}]: ").strip()
        if raw == '':
            return current
        try:
            val = float(raw)
        except ValueError:
            print("  Invalid number. Try again.")
            continue
        if lo is not None and val < lo:
            print(f"  Value must be >= {lo}")
            continue
        if hi is not None and val > hi:
            print(f"  Value must be <= {hi}")
            continue
        return val


def _read_int(prompt, current, lo=None, hi=None):
    """Prompt for an int, returning *current* on empty input."""
    while True:
        raw = input(f"  {prompt} [{current}]: ").strip()
        if raw == '':
            return current
        try:
            val = int(raw)
        except ValueError:
            print("  Invalid integer. Try again.")
            continue
        if lo is not None and val < lo:
            print(f"  Value must be >= {lo}")
            continue
        if hi is not None and val > hi:
            print(f"  Value must be <= {hi}")
            continue
        return val


def _read_bool(prompt, current):
    """Prompt for yes/no, returning *current* on empty input."""
    cur_str = "yes" if current else "no"
    raw = input(f"  {prompt} (yes/no) [{cur_str}]: ").strip().lower()
    if raw == '':
        return current
    return raw in ('y', 'yes', '1', 'true')


def _read_choice(prompt, options, current):
    """Prompt the user to pick from a numbered list of options."""
    print(f"  {prompt} (current: {current})")
    for i, opt in enumerate(options, 1):
        marker = " <--" if opt == current else ""
        print(f"    {i}. {opt}{marker}")
    while True:
        raw = input("  Choice: ").strip()
        if raw == '':
            return current
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("  Invalid choice. Try again.")


# ── sub-menus ────────────────────────────────────────────────────────

def _edit_platform(cfg):
    _clear()
    print("=" * 55)
    print("  Platform Parameters")
    print("=" * 55)
    p = cfg['platform']
    print()
    print("  Altitude: flight level of the ISR/NTISR platform")
    print("  above ground.")
    p['altitude_ft'] = _read_float("Altitude (ft)", p['altitude_ft'], lo=1000)
    print()
    print("  Speed: platform ground speed. Affects how long the")
    print("  platform remains within sensor range of the cell.")
    p['speed_kts'] = _read_float("Speed (kts)", p['speed_kts'], lo=0)
    print()
    print("  Heading: initial direction of flight (0 = North,")
    print("  90 = East).")
    p['heading_deg'] = _read_float("Heading (deg, 0=N)", p['heading_deg'])
    print()
    print("  Slant range: line-of-sight distance from the platform")
    print("  to the center of the search cell. Larger range means")
    print("  larger ground footprint but lower image quality.")
    cfg['geometry']['slant_range_nm'] = _read_float(
        "Slant range to cell (nm)", cfg['geometry']['slant_range_nm'], lo=0.1)


def _edit_cell_target(cfg):
    _clear()
    print("=" * 55)
    print("  Cell & Target Parameters")
    print("=" * 55)
    print()
    print("  Cell radius: the circular geographic area being")
    print("  searched. The target (car) stays inside this cell.")
    cfg['cell']['radius_nm'] = _read_float(
        "Cell radius (nm)", cfg['cell']['radius_nm'], lo=0.01)
    print()
    print("  Car speed: ground speed of the target vehicle.")
    print("  Faster cars are harder to detect because they cross")
    print("  the sensor footprint more quickly.")
    cfg['car']['speed_kts'] = _read_float(
        "Car speed (kts)", cfg['car']['speed_kts'], lo=0)
    print()
    print("  Heading-change interval: average time between random")
    print("  turns. Shorter intervals make the car path more")
    print("  erratic and harder to predict.")
    cfg['car']['heading_change_interval_s'] = _read_float(
        "Car heading-change interval (s)",
        cfg['car']['heading_change_interval_s'], lo=1)


def _edit_isr(cfg):
    _clear()
    print("=" * 55)
    print("  ISR Line Scanner (30 Hz raster)")
    print("=" * 55)
    print()
    print("  The ISR sensor uses a back-scan mirror to stabilise")
    print("  its line-of-sight, capturing one FOV-sized image per")
    print("  frame. It steps through a boustrophedon (lawnmower)")
    print("  raster covering the full cell.")
    print()
    s = cfg['isr']
    print("  FOV: instantaneous field of view per frame. Larger")
    print("  FOV covers more ground per frame but at lower")
    print("  resolution.")
    s['fov_deg'] = _read_float("Sensor FOV (deg)", s['fov_deg'], lo=0.01)
    print()
    print("  Frame rate: how many frames per second the sensor")
    print("  captures. Higher rate = faster cell coverage but each")
    print("  frame has shorter exposure time.")
    s['frame_rate_hz'] = _read_float("Frame rate (Hz)", s['frame_rate_hz'], lo=1)


def _edit_ntisr_ss(cfg):
    _clear()
    print("=" * 55)
    print("  NTISR Step-and-Stare")
    print("=" * 55)
    print()
    print("  The Step-and-Stare sensor points its narrow FOV at")
    print("  one position, dwells (stares) to build up detection")
    print("  probability, then slews (rotates) to the next")
    print("  position in a systematic raster pattern.")
    print()
    s = cfg['ntisr_step_stare']
    print("  FOV: the narrow targeting pod field of view. Smaller")
    print("  FOV = higher zoom / resolution but more positions")
    print("  needed to cover the cell.")
    s['fov_deg'] = _read_float("FOV (deg)", s['fov_deg'], lo=0.01)
    print()
    print("  Dwell time: how long the sensor stares at each")
    print("  position before moving on. Longer dwell increases")
    print("  detection probability at that spot but slows overall")
    print("  cell coverage.")
    s['dwell_time_s'] = _read_float("Dwell time (s)", s['dwell_time_s'], lo=0.1)
    print()
    print("  Slew rate: how fast the gimbal rotates between stare")
    print("  positions (degrees per second). Faster slew = less")
    print("  dead time between stares.")
    s['slew_rate_deg_s'] = _read_float(
        "Gimbal slew rate (deg/s)", s['slew_rate_deg_s'], lo=0.1)
    print()
    print("  Scan pattern: the order positions are visited.")
    print("    boustrophedon — row-by-row lawnmower pattern")
    print("    spiral — center-outward spiral")
    s['scan_pattern'] = _read_choice(
        "Scan pattern", ['boustrophedon', 'spiral'], s['scan_pattern'])


def _edit_ntisr_fmv(cfg):
    _clear()
    print("=" * 55)
    print("  NTISR FMV Search")
    print("=" * 55)
    print()
    print("  In FMV mode an operator watches a live video feed")
    print("  and manually pans the pod across the cell, biased")
    print("  toward roads and features where vehicles are likely")
    print("  to be found.")
    print()
    s = cfg['ntisr_fmv']
    print("  FOV: field of view of the targeting pod video.")
    s['fov_deg'] = _read_float("FOV (deg)", s['fov_deg'], lo=0.01)
    print()
    print("  Pan rate: maximum angular speed the operator slews")
    print("  the pod while searching.")
    s['slew_rate_deg_s'] = _read_float(
        "Operator pan rate (deg/s)", s['slew_rate_deg_s'], lo=0.1)
    print()
    print("  Search speed factor: fraction of max pan rate the")
    print("  operator actually uses (1.0 = slews at full speed,")
    print("  0.5 = half speed for more careful scanning).")
    s['search_speed_factor'] = _read_float(
        "Search speed factor (0-1)", s['search_speed_factor'], lo=0, hi=1)
    print()
    print("  Road bias: how strongly the operator's panning is")
    print("  attracted toward roads and linear features (0 = pure")
    print("  random search, 1.0 = always follows roads).")
    s['road_bias'] = _read_float(
        "Road/feature bias (0-1)", s['road_bias'], lo=0, hi=1)
    print()
    print("  Revisit tendency: likelihood the operator re-scans")
    print("  areas already visited instead of exploring new ground")
    print("  (0 = never revisits, 1.0 = always revisits).")
    s['revisit_tendency'] = _read_float(
        "Revisit tendency (0-1)", s['revisit_tendency'], lo=0, hi=1)


def _edit_detection(cfg):
    _clear()
    print("=" * 55)
    print("  Detection Model")
    print("=" * 55)
    print()
    print("  These parameters control how likely the sensor is")
    print("  to detect the target once it enters the field of view.")
    print()
    d = cfg['detection']
    print("  Base Pd: probability of detecting the target when it")
    print("  is centered in the FOV during one full dwell period.")
    print("  For ISR, this is the per-frame detection chance.")
    print("  For S&S/FMV, Pd accumulates over the dwell time up")
    print("  to this value.")
    d['pd_in_fov'] = _read_float(
        "Base Pd when in FOV (0-1)", d['pd_in_fov'], lo=0, hi=1)
    print()
    print("  Pd decay exponent: controls how rapidly detection")
    print("  probability drops off as the target moves away from")
    print("  boresight (center of FOV). Higher values mean Pd")
    print("  falls off more steeply toward the edges of the FOV.")
    print("  Formula: Pd * exp(-k * angle^2)")
    d['pd_decay_exponent'] = _read_float(
        "Pd off-boresight decay exponent", d['pd_decay_exponent'], lo=0)
    print()
    print("  Min dwell for detection: the minimum continuous time")
    print("  the target must remain in the FOV before a detection")
    print("  opportunity begins. Models the analyst's reaction")
    print("  time — they need a moment to recognise the target.")
    d['min_dwell_for_detect_s'] = _read_float(
        "Min dwell for detection (s)", d['min_dwell_for_detect_s'], lo=0)


def _edit_simulation(cfg):
    _clear()
    print("=" * 55)
    print("  Simulation Settings")
    print("=" * 55)
    s = cfg['simulation']
    print()
    print("  Duration: total simulated time per MC trial. Longer")
    print("  durations allow slower sensors more time to find the")
    print("  target but increase computation time.")
    s['duration_s'] = _read_float("Duration (s)", s['duration_s'], lo=1)
    print()
    print("  Time step: simulation tick interval. Smaller values")
    print("  give higher fidelity but run slower. 0.1s is a good")
    print("  balance for most scenarios.")
    s['time_step_s'] = _read_float("Time step (s)", s['time_step_s'], lo=0.01, hi=10)
    print()
    print("  MC trials: number of independent random trials.")
    print("  More trials give smoother statistics but take longer.")
    print("  500+ recommended for publication-quality results.")
    s['mc_trials'] = _read_int("Monte Carlo trials", s['mc_trials'], lo=1)
    print()
    print("  Random seed: fixes the random number generator for")
    print("  reproducible results. Use 'none' for a different")
    print("  outcome each run.")
    seed_raw = input(f"  Random seed (integer or 'none') [{s.get('random_seed', 42)}]: ").strip()
    if seed_raw.lower() == 'none':
        s['random_seed'] = None
    elif seed_raw != '':
        try:
            s['random_seed'] = int(seed_raw)
        except ValueError:
            print("  Invalid — keeping current seed.")


def _edit_output(cfg):
    _clear()
    print("=" * 55)
    print("  Output Settings")
    print("=" * 55)
    o = cfg['output']
    print()
    o['pdf_report'] = _read_bool("Generate PDF report?", o.get('pdf_report', True))
    print()
    o['animation'] = _read_bool("Generate animation frames?", o.get('animation', True))
    if o['animation']:
        print()
        print("  Animation speed factor: how much faster the video")
        print("  plays compared to real time (e.g. 5 = 5x speed-up).")
        o['animation_speed'] = _read_float(
            "Animation speed factor", o.get('animation_speed', 5.0), lo=0.1)
    print()
    raw = input(f"  Output directory [{o['output_dir']}]: ").strip()
    if raw:
        o['output_dir'] = raw


def _select_modes(selected):
    """Let the user toggle which sensor modes to simulate."""
    mode_labels = {
        'isr': 'ISR (Back-Scan Mirror)',
        'ntisr_ss': 'NTISR (Step & Stare)',
        'ntisr_fmv': 'NTISR (FMV Search)',
    }
    while True:
        _clear()
        print("=" * 55)
        print("  Select Sensor Modes to Simulate")
        print("=" * 55)
        all_modes = ['isr', 'ntisr_ss', 'ntisr_fmv']
        for i, m in enumerate(all_modes, 1):
            check = "[x]" if m in selected else "[ ]"
            print(f"  {i}. {check} {mode_labels[m]}")
        print()
        print("  Enter a number to toggle, A to select all, or Enter to confirm.")
        raw = input("  > ").strip().lower()
        if raw == '':
            if not selected:
                print("  You must select at least one mode.")
                _pause()
                continue
            return selected
        if raw == 'a':
            if set(selected) == set(all_modes):
                selected.clear()
            else:
                selected.clear()
                selected.extend(all_modes)
            continue
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(all_modes):
                m = all_modes[idx]
                if m in selected:
                    selected.remove(m)
                else:
                    selected.append(m)
        except ValueError:
            pass


def _show_summary(cfg):
    """Display current configuration summary inline."""
    from geometry import ground_footprint_at_cell, NM_TO_FT
    _clear()
    alt = cfg['platform']['altitude_ft']
    sr = cfg['geometry']['slant_range_nm']

    print("=" * 60)
    print("  Current Configuration Summary")
    print("=" * 60)
    print(f"  Platform :  {alt:.0f} ft AGL, {cfg['platform']['speed_kts']} kts, "
          f"heading {cfg['platform']['heading_deg']} deg")
    print(f"  Slant rng:  {sr} nm")
    print(f"  Cell     :  {cfg['cell']['radius_nm']} nm radius "
          f"({cfg['cell']['radius_nm'] * NM_TO_FT:.0f} ft)")
    print(f"  Car      :  {cfg['car']['speed_kts']} kts")
    print()

    sensors = [
        ("ISR FOV (1 frame)", cfg['isr']['fov_deg']),
        ("NTISR S&S FOV", cfg['ntisr_step_stare']['fov_deg']),
        ("NTISR FMV FOV", cfg['ntisr_fmv']['fov_deg']),
    ]
    print("  Ground footprints at cell center:")
    for name, fov in sensors:
        fp = ground_footprint_at_cell(alt, sr, fov)
        print(f"    {name:18s}: {fov:5.2f} deg -> {fp:.4f} nm ({fp*NM_TO_FT:.0f} ft)")

    print()
    print(f"  MC trials : {cfg['simulation']['mc_trials']}")
    print(f"  Duration  : {cfg['simulation']['duration_s']}s,  dt={cfg['simulation']['time_step_s']}s")
    print(f"  Pd in FOV : {cfg['detection']['pd_in_fov']}")
    print(f"  Output dir: {cfg['output']['output_dir']}")
    print("=" * 60)
    _pause()


def _save_config(cfg):
    """Save current config to a YAML file."""
    raw = input("  Save config as (filename, e.g. my_scenario.yaml): ").strip()
    if not raw:
        print("  Cancelled.")
        return
    if not raw.endswith(('.yaml', '.yml')):
        raw += '.yaml'
    with open(raw, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved to {os.path.abspath(raw)}")
    _pause()


def _load_config_from_file(cfg):
    """Load a YAML config file and merge into current cfg."""
    raw = input("  Config file path: ").strip()
    if not raw or not os.path.exists(raw):
        print("  File not found.")
        _pause()
        return cfg
    with open(raw, 'r') as f:
        user_cfg = yaml.safe_load(f)
    if user_cfg:
        _deep_update(cfg, user_cfg)
        print("  Config loaded and merged.")
    else:
        print("  File was empty — no changes.")
    _pause()
    return cfg


def _deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# ── ISR optimisation runner ──────────────────────────────────────────

def _run_optimization(cfg):
    """Run the ISR parameter sweep optimisation from the menu."""
    _clear()
    print("=" * 55)
    print("  ISR Scan Parameter Optimisation")
    print("=" * 55)
    print()
    print("  This sweeps ISR sensor and environmental parameters")
    print("  to find the combination that maximises detection.")
    print()
    print("  Grid options:")
    print("    Quick  — ~36 combinations  (a few minutes)")
    print("    Full   — ~1,080 combinations (may take a while)")
    print()

    print(f"  ISR FOV is fixed at {cfg['isr']['fov_deg']} deg (from config)")
    print()

    quick = True
    raw = input("  Grid mode (1=Quick, 2=Full) [1]: ").strip()
    if raw == '2':
        quick = False
    print()

    trials = _read_int("MC trials per point", 50, lo=5, hi=10000)
    top_n = _read_int("Top N results to display", 10, lo=1, hi=100)
    print()

    output_dir = cfg['output'].get('output_dir', './results')
    raw = input(f"  Output directory [{output_dir}]: ").strip()
    if raw:
        output_dir = raw
    print()

    # Run via the standalone optimize_isr_scan module
    print("  Starting optimisation...")
    print("  (this may take a while — progress is shown below)")
    print()

    try:
        from optimize_isr_scan import (load_base_config, build_sweep_grid,
                                       total_combinations, evaluate_point,
                                       write_csv, print_summary,
                                       generate_heatmap, format_eta)
        import itertools
        import time

        base_cfg = copy.deepcopy(cfg)
        grid = build_sweep_grid(quick=quick)
        n_total = total_combinations(grid)

        param_names = list(grid.keys())
        param_values = [grid[k] for k in param_names]

        grid_label = 'quick (coarse)' if quick else 'full'
        print(f"  Grid: {grid_label}, {n_total} combinations, "
              f"{trials} trials each")
        print(f"  ISR FOV: {cfg['isr']['fov_deg']} deg (fixed)")
        print()

        all_rows = []
        t_start = time.time()
        completed = 0
        skipped = 0

        for combo in itertools.product(*param_values):
            params = dict(zip(param_names, combo))
            completed += 1

            elapsed = time.time() - t_start
            if completed > 1:
                avg = elapsed / (completed - 1)
                eta = avg * (n_total - completed + 1)
            else:
                eta = float('inf')

            label = (f"FPS={params['frame_rate_hz']:.0f} "
                     f"R={params['cell_radius_nm']:.1f} "
                     f"SR={params['slant_range_nm']:.0f}")
            print(f"  [{completed}/{n_total}] {label} "
                  f"(elapsed {format_eta(elapsed)}, "
                  f"ETA {format_eta(eta)})", end="", flush=True)

            result = evaluate_point(base_cfg, params, trials)

            row = {
                "frame_rate_hz":      params["frame_rate_hz"],
                "cell_radius_nm":     params["cell_radius_nm"],
                "slant_range_nm":     params["slant_range_nm"],
                "altitude_ft":        params["altitude_ft"],
                "detection_rate_pct": round(result["detection_rate"] * 100, 2),
                "mean_ttfd_s":        round(result["mean_ttfd"], 3)
                                      if result["mean_ttfd"] != float('inf')
                                      else float("inf"),
                "scan_cycle_time_s":  round(result["scan_cycle_time"], 4)
                                      if result["scan_cycle_time"] != float('inf')
                                      else float("inf"),
                "skipped":            result["skipped"],
                "skip_reason":        result.get("skip_reason", ""),
            }
            all_rows.append(row)

            if result["skipped"]:
                skipped += 1
                print(f"  => SKIP ({result['skip_reason']})")
            else:
                det = f"{row['detection_rate_pct']:.1f}%"
                ttfd = (f"{row['mean_ttfd_s']:.1f}s"
                        if row['mean_ttfd_s'] != float('inf') else "N/A")
                print(f"  => Det={det}, TTFD={ttfd}")

        # Write CSV
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "isr_optimization.csv")
        write_csv(all_rows, csv_path)
        print(f"\n  CSV saved: {os.path.abspath(csv_path)}")

        # Print summary
        print_summary(all_rows, top_n=top_n)

        # Heatmap
        heatmap_path = os.path.join(output_dir,
                                    "isr_optimization_heatmap.pdf")
        print("  Generating heatmap...", end=" ", flush=True)
        if generate_heatmap(all_rows, heatmap_path):
            print(f"saved: {os.path.abspath(heatmap_path)}")
        else:
            print("skipped (matplotlib not available or no valid data)")

        total_time = time.time() - t_start
        print(f"\n  Total time: {format_eta(total_time)} "
              f"({completed} combos, {skipped} skipped)")

    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        import traceback
        traceback.print_exc()

    print()
    _pause()


# ── comprehensive report generator ──────────────────────────────────

def _generate_report(cfg):
    """Launch the comprehensive report generator from the menu."""
    _clear()
    print("=" * 55)
    print("  Comprehensive Report Generator")
    print("=" * 55)
    print()
    print("  Generates a multi-page PDF report including:")
    print("    - Simulation methodology (manual)")
    print("    - Sensor model descriptions")
    print("    - Detection & target model details")
    print("    - Geometry explanations")
    print("    - Configuration summary")
    print("    - ISR optimisation results + CSV")
    print("    - Sensitivity analysis charts + CSV")
    print()

    run_opt = _read_bool("Include ISR optimisation sweep?", True)

    opt_quick = True
    opt_trials = 50
    if run_opt:
        print()
        print("  Optimisation grid:")
        print("    Quick  — ~36 combinations  (faster)")
        print("    Full   — ~1,080 combinations (slower)")
        raw = input("  Grid mode (1=Quick, 2=Full) [1]: ").strip()
        if raw == '2':
            opt_quick = False
        opt_trials = _read_int("MC trials per optimisation point", 50,
                               lo=5, hi=10000)

    print()
    sens_trials = _read_int("MC trials per sensitivity point", 50,
                            lo=5, hi=10000)

    output_dir = cfg['output'].get('output_dir', './results')
    print()
    raw = input(f"  Output directory [{output_dir}]: ").strip()
    if raw:
        output_dir = raw

    print()
    print("  Starting report generation...")
    print("  (this will take a while — progress is shown below)")
    print()

    try:
        from report_generator import generate_comprehensive_report

        files = generate_comprehensive_report(
            cfg,
            output_dir=output_dir,
            run_optimization=run_opt,
            optimization_quick=opt_quick,
            optimization_trials=opt_trials,
            sensitivity_trials=sens_trials,
        )

        print()
        print("  " + "=" * 50)
        print("  Report generation complete!")
        print("  " + "=" * 50)
        print(f"  PDF report:      {os.path.abspath(files['pdf'])}")
        if 'optimization_csv' in files:
            print(f"  Optimisation CSV: "
                  f"{os.path.abspath(files['optimization_csv'])}")
        print(f"  Sensitivity CSV:  "
              f"{os.path.abspath(files['sensitivity_csv'])}")

    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        import traceback
        traceback.print_exc()

    print()
    _pause()


# ── main menu ────────────────────────────────────────────────────────

def interactive_menu(cfg):
    """
    Run the interactive menu loop.

    Args:
        cfg: Mutable configuration dict (modified in-place).

    Returns:
        (cfg, modes) on "Run", or None if the user quits.
    """
    selected_modes = ['isr', 'ntisr_ss', 'ntisr_fmv']

    while True:
        _clear()
        n_modes = len(selected_modes)
        print("=" * 55)
        print("  NTISR vs ISR — Interactive Menu")
        print("=" * 55)
        print()
        print("  Parameter Groups")
        print("  ─────────────────────────────────────────")
        print("  1.  Platform & Geometry")
        print("  2.  Cell & Target (car)")
        print("  3.  ISR Line Scanner (30 Hz raster)")
        print("  4.  NTISR Step-and-Stare")
        print("  5.  NTISR FMV Search")
        print("  6.  Detection Model")
        print("  7.  Simulation Settings")
        print("  8.  Output Settings")
        print()
        print("  Actions")
        print("  ─────────────────────────────────────────")
        print(f"  9.  Select sensor modes  ({n_modes} selected)")
        print("  10. View config summary")
        print("  11. Load config from file")
        print("  12. Save config to file")
        print()
        print("  Analysis & Reports")
        print("  ─────────────────────────────────────────")
        print("  13. Run ISR Optimisation (parameter sweep)")
        print("  14. Generate Comprehensive Report")
        print()
        print("  R.  Run simulation")
        print("  Q.  Quit")
        print()

        choice = input("  Enter choice: ").strip().lower()

        if choice == '1':
            _edit_platform(cfg)
        elif choice == '2':
            _edit_cell_target(cfg)
        elif choice == '3':
            _edit_isr(cfg)
        elif choice == '4':
            _edit_ntisr_ss(cfg)
        elif choice == '5':
            _edit_ntisr_fmv(cfg)
        elif choice == '6':
            _edit_detection(cfg)
        elif choice == '7':
            _edit_simulation(cfg)
        elif choice == '8':
            _edit_output(cfg)
        elif choice == '9':
            _select_modes(selected_modes)
        elif choice == '10':
            _show_summary(cfg)
        elif choice == '11':
            cfg = _load_config_from_file(cfg)
        elif choice == '12':
            _save_config(cfg)
        elif choice == '13':
            _run_optimization(cfg)
        elif choice == '14':
            _generate_report(cfg)
        elif choice == 'r':
            if not selected_modes:
                print("  No sensor modes selected. Pick at least one (option 9).")
                _pause()
                continue
            return cfg, selected_modes
        elif choice == 'q':
            return None
        else:
            print("  Invalid choice.")
            _pause()
