"""
Monte Carlo simulation engine.

Runs N trials for each sensor mode, collecting detection statistics:
- Time to first detection
- Cumulative probability of detection over time
- Number of detection events per trial
- Dwell-on-target statistics
"""

import numpy as np
from car_model import Car
from sensors import ISRBackScan, NTISRStepStare, NTISRFMV


class MCResult:
    """Container for Monte Carlo results of a single sensor mode."""

    def __init__(self, mode_name: str, n_trials: int, n_steps: int, dt: float):
        self.mode_name = mode_name
        self.n_trials = n_trials
        self.n_steps = n_steps
        self.dt = dt

        # Per-trial results
        self.time_to_first_detect = np.full(n_trials, np.inf)
        self.total_detections = np.zeros(n_trials, dtype=int)
        self.cumulative_pd = np.zeros((n_trials, n_steps))  # 1 if detected by step t

        # Aggregate (computed after all trials)
        self.mean_ttfd = 0.0
        self.median_ttfd = 0.0
        self.prob_detect_by_time = None  # CDF of first-detection time
        self.mean_cumulative_pd = None

    def finalize(self):
        """Compute aggregate statistics after all trials."""
        finite_ttfd = self.time_to_first_detect[
            np.isfinite(self.time_to_first_detect)]
        if len(finite_ttfd) > 0:
            self.mean_ttfd = np.mean(finite_ttfd)
            self.median_ttfd = np.median(finite_ttfd)
        else:
            self.mean_ttfd = np.inf
            self.median_ttfd = np.inf

        # CDF: fraction of trials where detection occurred by each time step
        self.mean_cumulative_pd = np.mean(self.cumulative_pd, axis=0)

        # Fraction of trials that detected at all
        self.overall_detect_fraction = np.mean(
            np.isfinite(self.time_to_first_detect))


def run_single_trial(sensor, car, dt: float, n_steps: int,
                     trial_idx: int, result: MCResult):
    """Run one MC trial: simulate car + sensor interaction."""
    detected = False

    for step_i in range(n_steps):
        t = step_i * dt

        # Advance car
        car.step(dt)

        # Advance sensor
        sensor.step(t, dt)

        # Check detection
        car_x, car_y = car.get_position()
        if sensor.check_detection(car_x, car_y, dt):
            result.total_detections[trial_idx] += 1
            if not detected:
                result.time_to_first_detect[trial_idx] = t
                detected = True

        # Mark cumulative: 1 from first detection onward
        if detected:
            result.cumulative_pd[trial_idx, step_i] = 1.0


def build_sensor(mode: str, cfg: dict, cell_cx: float, cell_cy: float,
                 rng: np.random.Generator):
    """Factory: build a sensor object from config."""
    altitude_ft = cfg['platform']['altitude_ft']
    slant_range_nm = cfg['geometry']['slant_range_nm']
    cell_radius_nm = cfg['cell']['radius_nm']
    pd = cfg['detection']['pd_in_fov']
    heading_rad = np.radians(cfg['platform']['heading_deg'])

    if mode == 'isr':
        return ISRBackScan(
            altitude_ft, slant_range_nm, cell_cx, cell_cy,
            cell_radius_nm, pd, rng,
            total_fov_deg=cfg['isr']['total_fov_deg'],
            ifov_deg=cfg['isr']['ifov_deg'],
            sweep_period_s=cfg['isr']['sweep_period_s'],
            along_track_fov_deg=cfg['isr']['along_track_fov_deg'],
            platform_heading_rad=heading_rad
        )
    elif mode == 'ntisr_ss':
        return NTISRStepStare(
            altitude_ft, slant_range_nm, cell_cx, cell_cy,
            cell_radius_nm, pd, rng,
            fov_deg=cfg['ntisr_step_stare']['fov_deg'],
            dwell_time_s=cfg['ntisr_step_stare']['dwell_time_s'],
            slew_rate_deg_s=cfg['ntisr_step_stare']['slew_rate_deg_s'],
            min_dwell_s=cfg['detection']['min_dwell_for_detect_s']
        )
    elif mode == 'ntisr_fmv':
        return NTISRFMV(
            altitude_ft, slant_range_nm, cell_cx, cell_cy,
            cell_radius_nm, pd, rng,
            fov_deg=cfg['ntisr_fmv']['fov_deg'],
            slew_rate_deg_s=cfg['ntisr_fmv']['slew_rate_deg_s'],
            search_speed_factor=cfg['ntisr_fmv']['search_speed_factor'],
            road_bias=cfg['ntisr_fmv']['road_bias'],
            revisit_tendency=cfg['ntisr_fmv']['revisit_tendency']
        )
    else:
        raise ValueError(f"Unknown sensor mode: {mode}")


def run_mc(cfg: dict, mode: str, progress_callback=None) -> MCResult:
    """
    Run full Monte Carlo simulation for one sensor mode.

    Args:
        cfg: Full configuration dict.
        mode: 'isr', 'ntisr_ss', or 'ntisr_fmv'.
        progress_callback: Optional callable(trial_idx, n_trials) for progress.

    Returns:
        MCResult with all statistics.
    """
    n_trials = cfg['simulation']['mc_trials']
    duration = cfg['simulation']['duration_s']
    dt = cfg['simulation']['time_step_s']
    n_steps = int(duration / dt)
    seed = cfg['simulation'].get('random_seed', None)

    cell_cx = 0.0  # Cell centered at origin
    cell_cy = 0.0
    cell_radius = cfg['cell']['radius_nm']

    result = MCResult(mode, n_trials, n_steps, dt)

    for trial in range(n_trials):
        trial_seed = (seed + trial * 1000) if seed is not None else None
        rng = np.random.default_rng(trial_seed)

        car = Car(
            cell_center_x=cell_cx,
            cell_center_y=cell_cy,
            cell_radius_nm=cell_radius,
            speed_kts=cfg['car']['speed_kts'],
            heading_change_interval_s=cfg['car']['heading_change_interval_s'],
            rng=rng
        )

        sensor = build_sensor(mode, cfg, cell_cx, cell_cy, rng)

        run_single_trial(sensor, car, dt, n_steps, trial, result)

        if progress_callback:
            progress_callback(trial + 1, n_trials)

    result.finalize()
    return result
