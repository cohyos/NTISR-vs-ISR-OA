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
    p['altitude_ft'] = _read_float("Altitude (ft)", p['altitude_ft'], lo=1000)
    p['speed_kts'] = _read_float("Speed (kts)", p['speed_kts'], lo=0)
    p['heading_deg'] = _read_float("Heading (deg, 0=N)", p['heading_deg'])
    cfg['geometry']['slant_range_nm'] = _read_float(
        "Slant range to cell (nm)", cfg['geometry']['slant_range_nm'], lo=0.1)


def _edit_cell_target(cfg):
    _clear()
    print("=" * 55)
    print("  Cell & Target Parameters")
    print("=" * 55)
    cfg['cell']['radius_nm'] = _read_float(
        "Cell radius (nm)", cfg['cell']['radius_nm'], lo=0.01)
    cfg['car']['speed_kts'] = _read_float(
        "Car speed (kts)", cfg['car']['speed_kts'], lo=0)
    cfg['car']['heading_change_interval_s'] = _read_float(
        "Car heading-change interval (s)",
        cfg['car']['heading_change_interval_s'], lo=1)


def _edit_isr(cfg):
    _clear()
    print("=" * 55)
    print("  ISR Sensor (Back-Scanning Mirror)")
    print("=" * 55)
    s = cfg['isr']
    s['total_fov_deg'] = _read_float("Total FOV (deg)", s['total_fov_deg'], lo=0.1)
    s['ifov_deg'] = _read_float("Instantaneous FOV (deg)", s['ifov_deg'], lo=0.01)
    s['sweep_period_s'] = _read_float("Sweep period (s)", s['sweep_period_s'], lo=0.1)
    s['along_track_fov_deg'] = _read_float(
        "Along-track FOV (deg)", s['along_track_fov_deg'], lo=0.01)


def _edit_ntisr_ss(cfg):
    _clear()
    print("=" * 55)
    print("  NTISR Step-and-Stare")
    print("=" * 55)
    s = cfg['ntisr_step_stare']
    s['fov_deg'] = _read_float("FOV (deg)", s['fov_deg'], lo=0.01)
    s['dwell_time_s'] = _read_float("Dwell time (s)", s['dwell_time_s'], lo=0.1)
    s['slew_rate_deg_s'] = _read_float(
        "Gimbal slew rate (deg/s)", s['slew_rate_deg_s'], lo=0.1)
    s['scan_pattern'] = _read_choice(
        "Scan pattern", ['boustrophedon', 'spiral'], s['scan_pattern'])


def _edit_ntisr_fmv(cfg):
    _clear()
    print("=" * 55)
    print("  NTISR FMV Search")
    print("=" * 55)
    s = cfg['ntisr_fmv']
    s['fov_deg'] = _read_float("FOV (deg)", s['fov_deg'], lo=0.01)
    s['slew_rate_deg_s'] = _read_float(
        "Operator pan rate (deg/s)", s['slew_rate_deg_s'], lo=0.1)
    s['search_speed_factor'] = _read_float(
        "Search speed factor (0-1)", s['search_speed_factor'], lo=0, hi=1)
    s['road_bias'] = _read_float(
        "Road/feature bias (0-1)", s['road_bias'], lo=0, hi=1)
    s['revisit_tendency'] = _read_float(
        "Revisit tendency (0-1)", s['revisit_tendency'], lo=0, hi=1)


def _edit_detection(cfg):
    _clear()
    print("=" * 55)
    print("  Detection Model")
    print("=" * 55)
    d = cfg['detection']
    d['pd_in_fov'] = _read_float(
        "Base Pd when in FOV (0-1)", d['pd_in_fov'], lo=0, hi=1)
    d['pd_decay_exponent'] = _read_float(
        "Pd off-boresight decay exponent", d['pd_decay_exponent'], lo=0)
    d['min_dwell_for_detect_s'] = _read_float(
        "Min dwell for detection (s)", d['min_dwell_for_detect_s'], lo=0)


def _edit_simulation(cfg):
    _clear()
    print("=" * 55)
    print("  Simulation Settings")
    print("=" * 55)
    s = cfg['simulation']
    s['duration_s'] = _read_float("Duration (s)", s['duration_s'], lo=1)
    s['time_step_s'] = _read_float("Time step (s)", s['time_step_s'], lo=0.01, hi=10)
    s['mc_trials'] = _read_int("Monte Carlo trials", s['mc_trials'], lo=1)
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
    o['pdf_report'] = _read_bool("Generate PDF report?", o.get('pdf_report', True))
    o['animation'] = _read_bool("Generate animation frames?", o.get('animation', True))
    if o['animation']:
        o['animation_speed'] = _read_float(
            "Animation speed factor", o.get('animation_speed', 5.0), lo=0.1)
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
        ("ISR total FOV", cfg['isr']['total_fov_deg']),
        ("ISR IFOV", cfg['isr']['ifov_deg']),
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
        print("  3.  ISR Sensor (back-scan mirror)")
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
