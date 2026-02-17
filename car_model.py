"""
Car (target) movement model inside a circular cell.

Implements a random-walk with reflective boundary:
- Car starts at a random position inside the cell.
- Car moves at a constant speed with periodic random heading changes.
- When the car hits the cell boundary it reflects inward.
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

        # Reflective boundary
        dist = np.sqrt((new_x - self.cx)**2 + (new_y - self.cy)**2)
        if dist > self.radius:
            # Reflect: reverse the component of velocity toward the boundary
            nx = (new_x - self.cx) / dist  # Normal pointing outward
            ny = (new_y - self.cy) / dist
            vx = self.speed_nm_s * np.cos(self.heading)
            vy = self.speed_nm_s * np.sin(self.heading)
            dot = vx * nx + vy * ny
            vx_ref = vx - 2 * dot * nx
            vy_ref = vy - 2 * dot * ny
            self.heading = np.arctan2(vy_ref, vx_ref)

            # Place car just inside boundary
            new_x = self.cx + nx * (self.radius - 0.001)
            new_y = self.cy + ny * (self.radius - 0.001)

        self.x = new_x
        self.y = new_y

    def get_position(self) -> tuple:
        return (self.x, self.y)
