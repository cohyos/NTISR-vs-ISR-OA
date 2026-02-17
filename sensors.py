"""
Sensor models for ISR, NTISR step-and-stare, and NTISR FMV modes.

Each sensor produces a time-varying footprint on the ground and evaluates
whether the target falls within the footprint at each time step.

Detection model per sensor type:
  - ISR: designed for sweep detection — each time the strip passes over the
    target there is an instantaneous Pd roll (no min dwell required).
  - NTISR S&S: target must be in the stare footprint during dwell — cumulative
    Pd grows with dwell time, requires min dwell threshold.
  - NTISR FMV: operator must have target in FOV long enough to recognize it —
    uses a recognition dwell threshold, then Pd per timestep.
"""

import numpy as np
from geometry import (ground_footprint_at_cell, point_in_rect_footprint,
                      KTS_TO_NM_PER_S, DEG_TO_RAD, FT_TO_NM)


class SensorBase:
    """Base class for all sensor modes."""

    def __init__(self, altitude_ft: float, slant_range_nm: float,
                 cell_center_x: float, cell_center_y: float,
                 cell_radius_nm: float, pd_in_fov: float,
                 rng: np.random.Generator):
        self.altitude_ft = altitude_ft
        self.slant_range_nm = slant_range_nm
        self.cx = cell_center_x
        self.cy = cell_center_y
        self.cell_radius = cell_radius_nm
        self.pd_in_fov = pd_in_fov
        self.rng = rng

        # Current footprint center on ground (nm, relative coords)
        self.fp_x = cell_center_x
        self.fp_y = cell_center_y
        self.fp_width = 0.0   # nm
        self.fp_height = 0.0  # nm
        self.fp_rotation = 0.0  # rad

        # Track continuous dwell time on target
        self.dwell_on_target = 0.0

    def reset(self):
        """Reset sensor state for a new MC trial."""
        self.dwell_on_target = 0.0

    def step(self, t: float, dt: float):
        """Update sensor pointing for current time. Override in subclass."""
        raise NotImplementedError

    def _target_in_footprint(self, car_x: float, car_y: float) -> bool:
        """Check if target is inside current footprint."""
        return point_in_rect_footprint(
            car_x, car_y,
            self.fp_x, self.fp_y,
            self.fp_width, self.fp_height,
            self.fp_rotation
        )

    def check_detection(self, car_x: float, car_y: float, dt: float) -> bool:
        """Check detection — override in subclass for mode-specific logic."""
        raise NotImplementedError

    def get_footprint(self) -> dict:
        """Return current footprint for visualization."""
        return {
            'x': self.fp_x, 'y': self.fp_y,
            'w': self.fp_width, 'h': self.fp_height,
            'rot': self.fp_rotation
        }


class ISRBackScan(SensorBase):
    """
    ISR sensor with back-scanning mirror.

    The mirror sweeps the full FOV back and forth continuously.
    At any instant the sensor sees a narrow strip (IFOV) within the total FOV.
    Detection model: each time the strip passes over the target, there is a
    single detection opportunity with Pd = pd_per_sweep. No minimum dwell
    needed — the system is designed for sweep detection.
    """

    def __init__(self, altitude_ft, slant_range_nm, cell_center_x,
                 cell_center_y, cell_radius_nm, pd_in_fov,
                 rng, total_fov_deg, ifov_deg, sweep_period_s,
                 along_track_fov_deg, platform_heading_rad):
        super().__init__(altitude_ft, slant_range_nm, cell_center_x,
                         cell_center_y, cell_radius_nm, pd_in_fov, rng)
        self.total_fov_deg = total_fov_deg
        self.ifov_deg = ifov_deg
        self.sweep_period_s = sweep_period_s
        self.along_track_fov_deg = along_track_fov_deg
        self.platform_heading = platform_heading_rad

        # Compute ground footprint dimensions
        self.total_cross_track_nm = ground_footprint_at_cell(
            altitude_ft, slant_range_nm, total_fov_deg)
        self.ifov_cross_track_nm = ground_footprint_at_cell(
            altitude_ft, slant_range_nm, ifov_deg)
        self.along_track_nm = ground_footprint_at_cell(
            altitude_ft, slant_range_nm, along_track_fov_deg)

        # The strip scans across the total FOV width
        self.scan_half_width = self.total_cross_track_nm / 2.0

        # Track whether target was in footprint last step (for edge detection)
        self._was_in_fp = False

    def reset(self):
        super().reset()
        self._was_in_fp = False

    def step(self, t: float, dt: float):
        """
        Update mirror position. Mirror sweeps as a triangle wave across
        the total FOV cross-track extent centered on cell center.
        """
        phase = (t % self.sweep_period_s) / self.sweep_period_s
        if phase < 0.5:
            offset = -self.scan_half_width + 2 * self.scan_half_width * (phase / 0.5)
        else:
            offset = self.scan_half_width - 2 * self.scan_half_width * ((phase - 0.5) / 0.5)

        # Offset is perpendicular to platform heading
        perp_angle = self.platform_heading + np.pi / 2
        self.fp_x = self.cx + offset * np.cos(perp_angle)
        self.fp_y = self.cy + offset * np.sin(perp_angle)
        self.fp_width = self.ifov_cross_track_nm
        self.fp_height = self.along_track_nm
        self.fp_rotation = self.platform_heading

    def check_detection(self, car_x: float, car_y: float, dt: float) -> bool:
        """
        ISR detection: each time the sweep strip enters the target position
        (rising edge), roll a Pd check. Also roll a per-timestep Pd while
        the strip overlaps the target.
        """
        in_fp = self._target_in_footprint(car_x, car_y)

        if in_fp:
            # Per-timestep detection opportunity while in IFOV
            # The strip crosses a point in roughly:
            #   t_cross = (ifov_width / total_width) * (sweep_period / 2)
            # Pd per crossing should equal pd_in_fov
            t_cross = (self.ifov_cross_track_nm / max(self.total_cross_track_nm, 0.001)) \
                      * (self.sweep_period_s / 2.0)
            if t_cross > 0:
                p_step = 1.0 - (1.0 - self.pd_in_fov) ** (dt / t_cross)
            else:
                p_step = self.pd_in_fov
            self._was_in_fp = True
            if self.rng.uniform() < p_step:
                return True
        else:
            self._was_in_fp = False
        return False


class NTISRStepStare(SensorBase):
    """
    NTISR sensor in step-and-stare mode.

    Narrow FOV steps through a boustrophedon (lawnmower) raster pattern
    across the cell, dwelling at each position before slewing to the next.

    Detection model: target must be inside the stare footprint. After a
    minimum dwell threshold, Pd accumulates per timestep. The cumulative
    Pd over one full dwell equals pd_in_fov.
    """

    def __init__(self, altitude_ft, slant_range_nm, cell_center_x,
                 cell_center_y, cell_radius_nm, pd_in_fov,
                 rng, fov_deg, dwell_time_s, slew_rate_deg_s,
                 min_dwell_s=0.5):
        super().__init__(altitude_ft, slant_range_nm, cell_center_x,
                         cell_center_y, cell_radius_nm, pd_in_fov, rng)
        self.fov_deg = fov_deg
        self.dwell_time_s = dwell_time_s
        self.slew_rate_deg_s = slew_rate_deg_s
        self.min_dwell_s = min_dwell_s

        # Ground footprint of the narrow FOV
        self.fp_size_nm = ground_footprint_at_cell(
            altitude_ft, slant_range_nm, fov_deg)
        self.fp_width = self.fp_size_nm
        self.fp_height = self.fp_size_nm

        # Build boustrophedon grid of stare positions covering the cell
        self.stare_positions = self._build_raster()
        self.current_idx = 0
        self.time_at_position = 0.0
        self.slewing = False
        self.slew_time_remaining = 0.0

    def _build_raster(self) -> list:
        """Build a boustrophedon scan pattern covering the circular cell."""
        positions = []
        step = self.fp_size_nm * 0.8  # 20% overlap between stares
        if step < 1e-6:
            step = 0.01
        r = self.cell_radius

        ny = int(2 * r / step) + 1
        nx = int(2 * r / step) + 1
        for iy in range(ny):
            y = self.cy - r + iy * step
            row_positions = []
            for ix in range(nx):
                x = self.cx - r + ix * step
                dx = x - self.cx
                dy = y - self.cy
                if dx**2 + dy**2 <= r**2:
                    row_positions.append((x, y))
            if iy % 2 == 1:
                row_positions.reverse()
            positions.extend(row_positions)

        if not positions:
            positions.append((self.cx, self.cy))
        return positions

    def reset(self):
        super().reset()
        self.current_idx = 0
        self.time_at_position = 0.0
        self.slewing = False
        self.slew_time_remaining = 0.0

    def step(self, t: float, dt: float):
        """Advance step-and-stare: dwell at position, then slew to next."""
        if self.slewing:
            self.slew_time_remaining -= dt
            if self.slew_time_remaining <= 0:
                self.slewing = False
                self.time_at_position = 0.0
            self.fp_width = 0.0
            self.fp_height = 0.0
            return

        self.time_at_position += dt
        pos = self.stare_positions[self.current_idx]
        self.fp_x = pos[0]
        self.fp_y = pos[1]
        self.fp_width = self.fp_size_nm
        self.fp_height = self.fp_size_nm
        self.fp_rotation = 0.0

        if self.time_at_position >= self.dwell_time_s:
            next_idx = (self.current_idx + 1) % len(self.stare_positions)
            cx, cy = self.stare_positions[self.current_idx]
            nx_, ny_ = self.stare_positions[next_idx]
            angular_dist = np.sqrt((nx_ - cx)**2 + (ny_ - cy)**2)
            alt_nm = self.altitude_ft * FT_TO_NM
            gr_nm = np.sqrt(max(self.slant_range_nm**2 - alt_nm**2, 0.01))
            angle_deg = np.degrees(np.arctan2(angular_dist, gr_nm))
            slew_time = angle_deg / self.slew_rate_deg_s if self.slew_rate_deg_s > 0 else 0

            self.current_idx = next_idx
            self.slewing = True
            self.slew_time_remaining = slew_time
            self.fp_width = 0.0
            self.fp_height = 0.0

    def check_detection(self, car_x: float, car_y: float, dt: float) -> bool:
        """
        Step-and-stare detection: target must be in FOV, and after minimum
        dwell the detection probability accumulates.
        """
        in_fp = self._target_in_footprint(car_x, car_y)
        if in_fp:
            self.dwell_on_target += dt
            if self.dwell_on_target >= self.min_dwell_s:
                # Pd is calibrated so that cumulative Pd over one full
                # dwell_time_s equals pd_in_fov
                effective_dwell = self.dwell_time_s - self.min_dwell_s
                if effective_dwell > 0:
                    p_step = 1.0 - (1.0 - self.pd_in_fov) ** (dt / effective_dwell)
                else:
                    p_step = self.pd_in_fov
                if self.rng.uniform() < p_step:
                    return True
        else:
            self.dwell_on_target = 0.0
        return False


class NTISRFMV(SensorBase):
    """
    NTISR sensor in Full Motion Video (FMV) search mode.

    Simulates an operator panning the pod across the cell, biased toward
    "road-like" corridors and points of interest.

    Detection model: operator must have target continuously in FOV for
    a recognition time (min_dwell_s), then Pd rolls per timestep.
    """

    def __init__(self, altitude_ft, slant_range_nm, cell_center_x,
                 cell_center_y, cell_radius_nm, pd_in_fov,
                 rng, fov_deg, slew_rate_deg_s, search_speed_factor,
                 road_bias, revisit_tendency, min_dwell_s=0.3):
        super().__init__(altitude_ft, slant_range_nm, cell_center_x,
                         cell_center_y, cell_radius_nm, pd_in_fov, rng)
        self.fov_deg = fov_deg
        self.slew_rate_deg_s = slew_rate_deg_s
        self.search_speed_factor = search_speed_factor
        self.road_bias = road_bias
        self.revisit_tendency = revisit_tendency
        self.min_dwell_s = min_dwell_s

        # Ground footprint
        self.fp_size_nm = ground_footprint_at_cell(
            altitude_ft, slant_range_nm, fov_deg)
        self.fp_width = self.fp_size_nm
        self.fp_height = self.fp_size_nm

        # Operator scan state
        self.scan_heading = rng.uniform(0, 2 * np.pi)
        self.time_to_heading_change = rng.exponential(5.0)

        # Convert slew rate to ground speed
        alt_nm = altitude_ft * FT_TO_NM
        gr_nm = np.sqrt(max(slant_range_nm**2 - alt_nm**2, 0.01))
        self.scan_speed_nm_s = (slew_rate_deg_s * DEG_TO_RAD * gr_nm
                                 * search_speed_factor)

        # Build simple "road network" as linear features through the cell
        self._build_roads()

        # Track visited regions for revisit logic
        self.visit_history = []

    def _build_roads(self):
        """Generate a simple random road network inside the cell."""
        n_roads = self.rng.integers(2, 5)
        self.roads = []
        for _ in range(n_roads):
            angle = self.rng.uniform(0, np.pi)
            offset_r = self.rng.uniform(0, self.cell_radius * 0.5)
            offset_a = self.rng.uniform(0, 2 * np.pi)
            mid_x = self.cx + offset_r * np.cos(offset_a)
            mid_y = self.cy + offset_r * np.sin(offset_a)
            self.roads.append((mid_x, mid_y, angle))

    def _nearest_road_attraction(self) -> tuple:
        """Compute a heading bias toward the nearest road."""
        if not self.roads:
            return (0.0, 0.0)

        best_dist = float('inf')
        best_along = (0.0, 0.0)
        for (rx, ry, ra) in self.roads:
            dx = self.fp_x - rx
            dy = self.fp_y - ry
            road_dir = np.array([np.cos(ra), np.sin(ra)])
            perp = np.array([-np.sin(ra), np.cos(ra)])
            perp_dist = abs(dx * perp[0] + dy * perp[1])
            if perp_dist < best_dist:
                best_dist = perp_dist
                along = road_dir
                toward = -np.sign(dx * perp[0] + dy * perp[1]) * perp
                best_along = (along[0] + 0.3 * toward[0],
                              along[1] + 0.3 * toward[1])

        norm = np.sqrt(best_along[0]**2 + best_along[1]**2)
        if norm > 0:
            return (best_along[0] / norm, best_along[1] / norm)
        return (0.0, 0.0)

    def reset(self):
        super().reset()
        self.scan_heading = self.rng.uniform(0, 2 * np.pi)
        self.time_to_heading_change = self.rng.exponential(5.0)
        self.visit_history = []
        self._build_roads()
        self.fp_x = self.cx
        self.fp_y = self.cy

    def step(self, t: float, dt: float):
        """Advance FMV operator scan — biased random walk."""
        self.time_to_heading_change -= dt
        if self.time_to_heading_change <= 0:
            rand_heading = self.rng.uniform(0, 2 * np.pi)
            rand_dir = np.array([np.cos(rand_heading), np.sin(rand_heading)])

            road_dir = self._nearest_road_attraction()

            blend_x = ((1.0 - self.road_bias) * rand_dir[0]
                       + self.road_bias * road_dir[0])
            blend_y = ((1.0 - self.road_bias) * rand_dir[1]
                       + self.road_bias * road_dir[1])

            self.scan_heading = np.arctan2(blend_y, blend_x)
            self.time_to_heading_change = self.rng.exponential(4.0)

        new_x = self.fp_x + self.scan_speed_nm_s * np.cos(self.scan_heading) * dt
        new_y = self.fp_y + self.scan_speed_nm_s * np.sin(self.scan_heading) * dt

        dist = np.sqrt((new_x - self.cx)**2 + (new_y - self.cy)**2)
        if dist > self.cell_radius - self.fp_size_nm / 2:
            to_center = np.arctan2(self.cy - new_y, self.cx - new_x)
            self.scan_heading = to_center + self.rng.uniform(-0.5, 0.5)
            new_x = self.fp_x + self.scan_speed_nm_s * np.cos(self.scan_heading) * dt
            new_y = self.fp_y + self.scan_speed_nm_s * np.sin(self.scan_heading) * dt

        self.fp_x = new_x
        self.fp_y = new_y
        self.fp_width = self.fp_size_nm
        self.fp_height = self.fp_size_nm
        self.fp_rotation = 0.0

        self.visit_history.append((self.fp_x, self.fp_y))

    def check_detection(self, car_x: float, car_y: float, dt: float) -> bool:
        """
        FMV detection: operator needs target in FOV for recognition time,
        then Pd accumulates. Recognition time is short (operator watching
        video continuously), so Pd builds quickly once target enters FOV.
        """
        in_fp = self._target_in_footprint(car_x, car_y)
        if in_fp:
            self.dwell_on_target += dt
            if self.dwell_on_target >= self.min_dwell_s:
                # Calibrate so that 2s of dwell gives cumulative Pd = pd_in_fov
                recognition_window = 2.0
                p_step = 1.0 - (1.0 - self.pd_in_fov) ** (dt / recognition_window)
                if self.rng.uniform() < p_step:
                    return True
        else:
            self.dwell_on_target = 0.0
        return False
