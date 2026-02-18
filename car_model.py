"""
Car (target) movement model inside a circular cell.

Implements a random-walk with reflective boundary:
- Car starts at a random position inside the cell.
- Car moves at a constant speed with periodic random heading changes.
- When the car hits the cell boundary it bounces back inward.
"""

import numpy as np
from geometry import KTS_TO_NM_PER_S


class Car:
    """Moving ground target inside a circular cell."""

    def __init__(self, cell_center_x: float, cell_center_y: float,
                 cell_radius_nm: float, speed_kts: float,
                 heading_change_interval_s: float, rng: np.random.Generator):
        self.cx = cell_center_x
        self.cy = cell_center_y
        self.radius = cell_radius_nm
        self.speed_nm_s = speed_kts * KTS_TO_NM_PER_S
        self.heading_interval = heading_change_interval_s
        self.rng = rng

        # Random initial position inside cell
        r = self.radius * np.sqrt(self.rng.uniform(0, 1))
        theta = self.rng.uniform(0, 2 * np.pi)
        self.x = self.cx + r * np.cos(theta)
        self.y = self.cy + r * np.sin(theta)

        # Random initial heading (radians, 0 = East)
        self.heading = self.rng.uniform(0, 2 * np.pi)

        # Time until next heading change
        self.time_to_change = self.rng.exponential(self.heading_interval)

    def step(self, dt: float):
        """Advance car by one time step."""
        # Update heading change timer
        self.time_to_change -= dt
        if self.time_to_change <= 0:
            self.heading = self.rng.uniform(0, 2 * np.pi)
            self.time_to_change = self.rng.exponential(self.heading_interval)

        # Move
        dx = self.speed_nm_s * np.cos(self.heading) * dt
        dy = self.speed_nm_s * np.sin(self.heading) * dt
        new_x = self.x + dx
        new_y = self.y + dy

        # Reflective boundary: on hitting the cell edge, pick a random
        # inward heading within a ±45° cone of the inward normal.
        # This avoids the tangential sliding that specular reflection
        # causes on grazing hits against a circular boundary.
        dist = np.sqrt((new_x - self.cx)**2 + (new_y - self.cy)**2)
        if dist > self.radius:
            # Unit outward normal at the exit point
            nx = (new_x - self.cx) / dist
            ny = (new_y - self.cy) / dist

            # Random heading within ±30° of the inward normal
            inward_angle = np.arctan2(-ny, -nx)
            jitter = self.rng.uniform(-np.pi / 6, np.pi / 6)
            self.heading = inward_angle + jitter

            # Place car at boundary contact point and step inward
            overshoot = dist - self.radius
            contact_x = self.cx + nx * self.radius
            contact_y = self.cy + ny * self.radius
            new_x = contact_x + np.cos(self.heading) * overshoot
            new_y = contact_y + np.sin(self.heading) * overshoot

            # Safety clamp: if still outside, pull inside
            dist2 = np.sqrt((new_x - self.cx)**2 + (new_y - self.cy)**2)
            if dist2 >= self.radius:
                scale = (self.radius * 0.90) / max(dist2, 1e-12)
                new_x = self.cx + (new_x - self.cx) * scale
                new_y = self.cy + (new_y - self.cy) * scale

            # Reset heading change timer so the car keeps this inward
            # heading long enough to move well away from the edge.
            self.time_to_change = self.rng.exponential(self.heading_interval)

        self.x = new_x
        self.y = new_y

    def get_position(self) -> tuple:
        return (self.x, self.y)
