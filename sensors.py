"""
Sensor models for ISR, NTISR step-and-stare, and NTISR FMV modes.

Each sensor produces a time-varying footprint on the ground and evaluates
whether the target falls within the footprint at each time step.

Detection model per sensor type:
  - ISR line scanner: 1-degree FOV steps through a boustrophedon raster at
    30 Hz.  The back-scan mirror only stabilises the LOS during each 1/30 s
    frame — it does NOT sweep across a wider FOV.  Each frame where the target
    is inside the FOV is an independent detection opportunity.
  - NTISR S&S: target must be in the stare footprint during dwell — cumulative
    Pd grows with dwell time, requires min dwell threshold.
  - NTISR FMV: same 1-degree FOV, but the operator slews the LOS manually.
    Operator must keep the target in FOV long enough for recognition, then
    Pd accumulates per timestep.
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

        # Track cumulative area swept (nm^2) for efficiency metrics
        self._area_swept = 0.0
        self._cell_area = np.pi * cell_radius_nm ** 2

    def reset(self):
        """Reset sensor state for a new MC trial."""
        self.dwell_on_target = 0.0
        self._area_swept = 0.0

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

    @property
    def coverage_ratio(self) -> float:
        """Total area swept divided by cell area."""
        if self._cell_area > 0:
            return self._area_swept / self._cell_area
        return 0.0

    def get_footprint(self) -> dict:
        """Return current footprint for visualization."""
        return {
            'x': self.fp_x, 'y': self.fp_y,
            'w': self.fp_width, 'h': self.fp_height,
            'rot': self.fp_rotation
        }


class ISRLineScan(SensorBase):
    """
    ISR line-scanner with back-scan stabilisation.

    The sensor has a fixed FOV (typically 1 degree).  It steps through a
    boustrophedon (row-by-row) raster covering the full cell at the
    configured frame rate (default 30 Hz).  The back-scan mirror stabilises
    the LOS during each frame so the image is smear-free, but it does NOT
    widen the instantaneous field — the sensor sees exactly one FOV-sized
    patch per frame.

    Detection model: each frame where the target is inside the FOV is an
    independent detection opportunity with probability ``pd_in_fov``.
    No minimum dwell is required; even a single 1/30 s frame is enough
    for a detection roll.
    """

    def __init__(self, altitude_ft, slant_range_nm, cell_center_x,
                 cell_center_y, cell_radius_nm, pd_in_fov,
                 rng, fov_deg, frame_rate_hz):
        super().__init__(altitude_ft, slant_range_nm, cell_center_x,
                         cell_center_y, cell_radius_nm, pd_in_fov, rng)
        self.fov_deg = fov_deg
        self.frame_rate_hz = frame_rate_hz
        self.frame_period = 1.0 / frame_rate_hz  # seconds per frame

        # Ground footprint of one FOV (square)
        self.fp_size_nm = ground_footprint_at_cell(
            altitude_ft, slant_range_nm, fov_deg)
        self.fp_width = self.fp_size_nm
        self.fp_height = self.fp_size_nm

        # Build boustrophedon raster covering full cell
        self.scan_positions = self._build_raster()
        self.current_idx = 0
        self.time_in_frame = 0.0

    def _build_raster(self) -> list:
        """Build a boustrophedon raster covering the circular cell."""
        positions = []
        step = self.fp_size_nm * 0.8  # 20 % overlap between frames
        if step < 1e-6:
            step = 0.01
        r = self.cell_radius

        ny = int(2 * r / step) + 1
        nx = int(2 * r / step) + 1
        for iy in range(ny):
            y = self.cy - r + iy * step
            row = []
            for ix in range(nx):
                x = self.cx - r + ix * step
                if (x - self.cx) ** 2 + (y - self.cy) ** 2 <= r ** 2:
                    row.append((x, y))
            if iy % 2 == 1:
                row.reverse()
            positions.extend(row)

        if not positions:
            positions.append((self.cx, self.cy))
        return positions

    @property
    def scan_cycle_time(self) -> float:
        """Time to complete one full raster of the cell (seconds)."""
        return len(self.scan_positions) * self.frame_period

    def reset(self):
        super().reset()
        self.current_idx = 0
        self.time_in_frame = 0.0

    def step(self, t: float, dt: float):
        """Advance the raster.  Each frame lasts 1/frame_rate_hz seconds."""
        self.time_in_frame += dt

        # Advance to next position(s) if frame period elapsed
        prev_idx = self.current_idx
        while self.time_in_frame >= self.frame_period:
            self.time_in_frame -= self.frame_period
            self.current_idx = (self.current_idx + 1) % len(self.scan_positions)
            # Each new raster position sweeps one footprint of area
            # (with 20% overlap, effective new area is ~64% of fp^2)
            self._area_swept += self.fp_size_nm ** 2 * 0.64

        pos = self.scan_positions[self.current_idx]
        self.fp_x = pos[0]
        self.fp_y = pos[1]
        self.fp_width = self.fp_size_nm
        self.fp_height = self.fp_size_nm
        self.fp_rotation = 0.0

    def check_detection(self, car_x: float, car_y: float, dt: float) -> bool:
        """
        Per-frame detection: if the target is inside the FOV during this
        simulation step, roll a single Pd check.  Because the sensor hops
        to a new position every 1/30 s, the target is only "seen" in the
        frames where the raster happens to cover its location.
        """
        if self._target_in_footprint(car_x, car_y):
            if self.rng.uniform() < self.pd_in_fov:
                return True
        return False


# Keep legacy alias so existing imports still work
ISRBackScan = ISRLineScan


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
        # Randomize start position so we don't always begin at the cell edge
        self.current_idx = int(rng.integers(0, len(self.stare_positions)))
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

            # Each new stare position sweeps one footprint of area
            self._area_swept += self.fp_size_nm ** 2 * 0.64

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
        """
        Compute a heading bias toward/along the nearest road.

        Roads are bidirectional — the along-road direction is chosen to
        be the one closer to the operator's current heading so the
        operator follows roads naturally rather than always being pulled
        in a single fixed direction.
        """
        if not self.roads:
            return (0.0, 0.0)

        best_dist = float('inf')
        best_along = (0.0, 0.0)
        heading_vec = np.array([np.cos(self.scan_heading),
                                np.sin(self.scan_heading)])

        for (rx, ry, ra) in self.roads:
            dx = self.fp_x - rx
            dy = self.fp_y - ry
            road_dir = np.array([np.cos(ra), np.sin(ra)])
            perp = np.array([-np.sin(ra), np.cos(ra)])
            perp_dist = abs(dx * perp[0] + dy * perp[1])
            if perp_dist < best_dist:
                best_dist = perp_dist
                # Pick the along-road direction closer to current heading
                if np.dot(road_dir, heading_vec) < 0:
                    along = -road_dir
                else:
                    along = road_dir
                toward = -np.sign(dx * perp[0] + dy * perp[1]) * perp
                best_along = (along[0] + 0.3 * toward[0],
                              along[1] + 0.3 * toward[1])

        norm = np.sqrt(best_along[0]**2 + best_along[1]**2)
        if norm > 0:
            return (best_along[0] / norm, best_along[1] / norm)
        return (0.0, 0.0)

    def _revisit_repulsion(self) -> tuple:
        """
        Compute a repulsion vector away from recently visited positions.

        Encourages the operator to explore unvisited areas rather than
        re-scanning the same patch.  Strength is controlled by
        revisit_tendency (0 = strong avoidance, 1 = no avoidance).
        """
        if not self.visit_history or self.revisit_tendency >= 1.0:
            return (0.0, 0.0)

        # Sample recent positions (last ~5 seconds of history)
        recent = self.visit_history[-50:]
        repel_x, repel_y = 0.0, 0.0
        for (vx, vy) in recent:
            dx = self.fp_x - vx
            dy = self.fp_y - vy
            d2 = dx * dx + dy * dy
            if d2 > 1e-8:
                # Inverse-distance repulsion (capped to avoid singularity)
                strength = 1.0 / max(d2, self.fp_size_nm ** 2)
                repel_x += dx * strength
                repel_y += dy * strength

        norm = np.sqrt(repel_x ** 2 + repel_y ** 2)
        if norm > 0:
            return (repel_x / norm, repel_y / norm)
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
        boundary_limit = self.cell_radius - self.fp_size_nm / 2
        if boundary_limit < 0.1 * self.cell_radius:
            boundary_limit = 0.1 * self.cell_radius

        # ── Heading update ─────────────────────────────────────────
        self.time_to_heading_change -= dt
        if self.time_to_heading_change <= 0:
            rand_heading = self.rng.uniform(0, 2 * np.pi)
            rand_dir = np.array([np.cos(rand_heading), np.sin(rand_heading)])

            road_dir = self._nearest_road_attraction()

            blend_x = ((1.0 - self.road_bias) * rand_dir[0]
                       + self.road_bias * road_dir[0])
            blend_y = ((1.0 - self.road_bias) * rand_dir[1]
                       + self.road_bias * road_dir[1])

            # Revisit avoidance: blend in repulsion from recently
            # visited areas so the operator explores new ground.
            repel = self._revisit_repulsion()
            avoid_w = (1.0 - self.revisit_tendency) * 0.3
            blend_x = (1.0 - avoid_w) * blend_x + avoid_w * repel[0]
            blend_y = (1.0 - avoid_w) * blend_y + avoid_w * repel[1]

            self.scan_heading = np.arctan2(blend_y, blend_x)
            self.time_to_heading_change = self.rng.exponential(4.0)

        # ── Move ───────────────────────────────────────────────────
        new_x = self.fp_x + self.scan_speed_nm_s * np.cos(self.scan_heading) * dt
        new_y = self.fp_y + self.scan_speed_nm_s * np.sin(self.scan_heading) * dt

        # ── Boundary reflection ────────────────────────────────────
        dist = np.sqrt((new_x - self.cx)**2 + (new_y - self.cy)**2)
        if dist > boundary_limit:
            nx = (new_x - self.cx) / dist
            ny = (new_y - self.cy) / dist

            # Specular reflection
            vx = np.cos(self.scan_heading)
            vy = np.sin(self.scan_heading)
            dot = vx * nx + vy * ny
            vx_ref = vx - 2 * dot * nx
            vy_ref = vy - 2 * dot * ny

            # For near-grazing hits (dot close to 0) the reflected
            # heading is nearly tangent to the boundary, which causes
            # the FMV to slide along the edge.  Blend toward the
            # inward normal so the operator bounces back into the
            # interior.  The blend weight is strongest for grazing
            # hits and zero for head-on hits.
            graze = 1.0 - min(abs(dot), 1.0)  # 1=grazing, 0=head-on
            inward_x = -nx
            inward_y = -ny
            inward_w = graze * 0.7
            vx_final = (1.0 - inward_w) * vx_ref + inward_w * inward_x
            vy_final = (1.0 - inward_w) * vy_ref + inward_w * inward_y
            self.scan_heading = np.arctan2(vy_final, vx_final)

            # Bounce the overshoot inward from the contact point
            overshoot = dist - boundary_limit
            contact_x = self.cx + nx * boundary_limit
            contact_y = self.cy + ny * boundary_limit
            new_x = contact_x + np.cos(self.scan_heading) * overshoot
            new_y = contact_y + np.sin(self.scan_heading) * overshoot

            # Safety clamp if still outside (near-tangent hit)
            dist2 = np.sqrt((new_x - self.cx)**2 + (new_y - self.cy)**2)
            if dist2 >= boundary_limit:
                scale = (boundary_limit * 0.90) / max(dist2, 1e-12)
                new_x = self.cx + (new_x - self.cx) * scale
                new_y = self.cy + (new_y - self.cy) * scale

            # Force an early heading change so the operator picks a
            # new search direction after bouncing off the edge
            self.time_to_heading_change = min(
                self.time_to_heading_change,
                self.rng.uniform(0.3, 1.0))

        # Track area swept: footprint width × distance moved
        dist_moved = np.sqrt((new_x - self.fp_x)**2 + (new_y - self.fp_y)**2)
        self._area_swept += self.fp_size_nm * dist_moved

        self.fp_x = new_x
        self.fp_y = new_y
        self.fp_width = self.fp_size_nm
        self.fp_height = self.fp_size_nm
        self.fp_rotation = 0.0

        self.visit_history.append((self.fp_x, self.fp_y))
        # Cap history to last ~10 seconds to avoid unbounded growth
        max_history = int(10.0 / max(dt, 0.01))
        if len(self.visit_history) > max_history:
            self.visit_history = self.visit_history[-max_history:]

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
