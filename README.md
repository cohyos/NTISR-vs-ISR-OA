# NTISR vs ISR Operational Analysis

**Monte Carlo comparison of three airborne EO/IR sensor search modes against a
moving ground target.**

This simulation tool evaluates the detection performance of ISR (Intelligence,
Surveillance, and Reconnaissance) and NTISR (Non-Traditional ISR) sensor
configurations when searching for moving vehicles inside a circular
geographic cell.  It produces statistical metrics (detection rate, time to
first detection, search efficiency), PDF reports, optimisation sweeps,
sensitivity analyses, and advanced multi-dimensional comparisons.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation & Requirements](#installation--requirements)
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)
5. [Simulation Flow](#simulation-flow)
6. [Sensor Models](#sensor-models)
   - [ISR Line Scanner (Back-Scan Mirror)](#1-isr-line-scanner-back-scan-mirror)
   - [NTISR Step-and-Stare](#2-ntisr-step-and-stare)
   - [NTISR Full Motion Video (FMV)](#3-ntisr-full-motion-video-fmv)
7. [Target (Car) Model](#target-car-model)
8. [Geometry & Physics](#geometry--physics)
9. [Efficiency Metrics](#efficiency-metrics)
10. [Detection Probability Theory](#detection-probability-theory)
    - [Cookie-Cutter Detection Model](#cookie-cutter-detection-model)
    - [Cumulative Detection Probability](#cumulative-detection-probability)
    - [Operator Recognition Model (FMV)](#operator-recognition-model-fmv)
    - [Calibration Method](#calibration-method)
11. [Mathematical Formulas & Models](#mathematical-formulas--models)
    - [Slant-Range Geometry](#slant-range-geometry)
    - [FOV-to-Footprint Projection](#fov-to-footprint-projection)
    - [Boustrophedon Scan Pattern](#boustrophedon-scan-pattern)
    - [Reflective Boundary Motion](#reflective-boundary-motion)
    - [Per-Step Detection Probability](#per-step-detection-probability)
    - [Monte Carlo Aggregation](#monte-carlo-aggregation)
12. [Configuration Reference](#configuration-reference)
13. [ISR Optimisation](#isr-optimisation)
14. [Advanced Analysis](#advanced-analysis)
    - [Duration Crossover Analysis](#duration-crossover-analysis)
    - [Multi-Platform Comparison](#multi-platform-comparison)
    - [Multi-Target Scenarios](#multi-target-scenarios)
    - [Moving Platform (Orbit) Analysis](#moving-platform-orbit-analysis)
    - [Parallel MC Execution](#parallel-mc-execution)
    - [Per-Trial Raw Data Export](#per-trial-raw-data-export)
15. [Comprehensive Report](#comprehensive-report)
16. [Output Files](#output-files)
17. [References & Sources](#references--sources)

---

## Overview

The tool compares three sensor modes under identical conditions:

| Mode | Description | Scan Strategy |
|------|-------------|---------------|
| **ISR Line Scanner** | Wide-area surveillance with back-scan stabilisation | Systematic boustrophedon raster at 30 Hz |
| **NTISR Step-and-Stare** | Narrow-FOV targeting pod, systematic raster | Dwell at each position, slew to next |
| **NTISR FMV** | Operator-guided full-motion-video search | Biased random walk toward road features |

All three modes search for one or more moving vehicles inside a circular cell.
The platform can be modelled as stationary or flying a racetrack orbit with
time-varying slant range.

**Key performance metrics:**

- **Detection rate**: fraction of Monte Carlo trials where at least one target
  was detected
- **Mean / Median TTFD**: Time To First Detection — how quickly the sensor
  finds the target
- **Cumulative Pd(t)**: probability that the target has been detected by time t
- **DTI**: Detection Timeliness Index — normalised AUC of the cumulative Pd
  curve (1.0 = instant detection, 0.0 = never detected)
- **Coverage ratio**: total area swept by the sensor / cell area
- **SEI**: Search Efficiency Index = DTI / Coverage ratio — measures how
  efficiently the sensor converts area coverage into detections

---

## Installation & Requirements

```
Python >= 3.8
numpy
matplotlib
PyYAML
```

Install dependencies:

```bash
pip install numpy matplotlib pyyaml
```

No additional compilation or external services are required.

---

## Quick Start

```bash
# Run with default parameters (CLI mode)
python main.py

# Launch interactive menu
python main.py --interactive

# Custom parameters
python main.py --altitude 25000 --trials 1000 --car-speed 60

# Run ISR optimisation sweep
python optimize_isr_scan.py --quick

# Generate comprehensive report
python report_generator.py

# Run advanced analyses (interactive menu option 15)
# Includes: duration crossover, multi-platform, multi-target, orbit analysis
python main.py --interactive    # then select option 15
```

---

## Architecture

```
main.py                 # Entry point — CLI and interactive launcher
interactive_menu.py     # Numbered-menu interactive parameter editor
config_default.yaml     # Default configuration (all parameters)
mc_engine.py            # Monte Carlo engine — run_mc(), MCResult
sensors.py              # Sensor classes: ISRLineScan, NTISRStepStare, NTISRFMV
car_model.py            # Target movement model with reflective boundary
geometry.py             # Slant-range geometry, FOV-to-footprint, coordinate utils
visualization.py        # PDF report generation, animation frames
optimize_isr_scan.py    # ISR parameter sweep optimizer
report_generator.py     # Comprehensive report with manual, optimization, sensitivity
advanced_analysis.py    # Advanced analyses: crossover, multi-platform, multi-target, orbit
```

---

## Simulation Flow

For each sensor mode the Monte Carlo engine executes the following loop:

```
FOR each trial in 1..N:
    1. Place car at random position inside cell (uniform in disk)
    2. Assign random heading to car
    3. Initialise sensor scan pattern

    FOR each time step t = 0, dt, 2*dt, ... , T:
        a. Advance car position (constant speed + reflective boundary)
        b. Update sensor pointing / footprint position
        c. IF car is inside sensor footprint:
             Apply mode-specific probabilistic detection model
             IF detection occurs: record TTFD, increment count
    END FOR

    Record trial results (TTFD, total detections, cumulative Pd)
END FOR

Compute aggregate statistics across all trials
```

The time step `dt` (default 0.1 s) determines the temporal resolution.  Each
trial is seeded deterministically: `trial_seed = base_seed + trial_index * 1000`.

---

## Sensor Models

### 1. ISR Line Scanner (Back-Scan Mirror)

**Physical description:**  The ISR sensor has a fixed angular field of view
(default 1.0 deg).  A back-scan mirror stabilises the line-of-sight during each
frame, producing a smear-free image.  The mirror does **not** widen the
instantaneous FOV — the sensor sees exactly one FOV-sized square patch per
frame.

**Scan pattern:**  The sensor steps through a **boustrophedon** (serpentine /
lawnmower) raster covering the full circular cell.  Scan positions are laid out
on a square grid with step size = 80% of the ground footprint (20% overlap).
Only positions whose centres fall within the cell circle are retained.  Odd-
numbered rows are traversed in reverse to create the serpentine path.

**Timing:**  The sensor hops to the next raster position every `1 / frame_rate_hz`
seconds (default 1/30 s).  The **scan cycle time** is the product of the total
number of raster positions and the frame period.

**Detection model:**  Each frame where the target is inside the FOV footprint
is an independent Bernoulli trial with success probability `pd_in_fov`.  No
minimum dwell is required — even a single 1/30 s frame suffices for a detection
opportunity.

> **Implementation:** `sensors.py → ISRLineScan`

### 2. NTISR Step-and-Stare

**Physical description:**  A narrow-FOV targeting pod (default 0.5 deg) that
dwells at each position to accumulate detection probability before slewing to
the next.

**Scan pattern:**  Same boustrophedon grid as ISR but using the NTISR FOV.  The
pod starts at a random grid position (uniform draw) to avoid bias from always
beginning at the cell edge.

**Timing:**  At each position the sensor dwells for `dwell_time_s` (default
2.0 s).  It then computes the angular distance to the next grid position and
slews at `slew_rate_deg_s` (default 20 deg/s).  During the slew the footprint
is set to zero (no detection possible).

**Detection model:**  The target must be continuously inside the FOV.  After
a minimum dwell threshold (`min_dwell_for_detect_s`, default 0.5 s), detection
probability accumulates per time step using the calibrated formula (see
[Cumulative Detection Probability](#cumulative-detection-probability) below).
If the target leaves the FOV, the dwell timer resets to zero.

> **Implementation:** `sensors.py → NTISRStepStare`

### 3. NTISR Full Motion Video (FMV)

**Physical description:**  An operator watches live video from the targeting
pod and manually pans across the cell, biased toward roads and linear features.

**Operator model:**  The scan path is a **biased random walk**:

1. A random heading is drawn at exponential intervals (mean ~4 s).
2. The heading is blended with an attraction vector toward the nearest road
   feature, weighted by `road_bias` (default 0.7).
3. The pod moves at `ground_speed = slew_rate * ground_range * search_speed_factor`.
4. When the footprint approaches the cell boundary the operator reflects
   inward with a small random jitter.

**Road network:**  Each trial generates 2–5 random linear "road" features
through the cell.  Each road is parameterised by a midpoint and angle.  The
nearest-road attraction blends an along-road component with a toward-road
component (30% of the attraction force).

**Detection model:**  The operator must keep the target continuously in the FOV
for a recognition time (`min_dwell_s`, default 0.3 s).  After recognition,
detection probability accumulates over a 2 s window:

```
p_step = 1 - (1 - pd_in_fov)^(dt / 2.0)
```

> **Implementation:** `sensors.py → NTISRFMV`

---

## Target (Car) Model

The ground target is a vehicle moving inside the circular search cell:

- **Initial position:** uniform random distribution inside the circle
  `r = R * sqrt(U(0,1))`, `theta = U(0, 2*pi)`
- **Initial heading:** uniform random in `[0, 2*pi)`
- **Speed:** constant (default 30 kts = 0.00833 nm/s)
- **Heading changes:** exponentially distributed intervals (mean =
  `heading_change_interval_s`, default 15 s); new heading drawn uniformly
- **Reflective boundary:** when the car would exit the cell, a new heading is
  drawn from a ±30° inward cone centred on the boundary inward normal.  The
  heading-change timer is reset to prevent the car from immediately turning
  back toward the edge.  This avoids the tangential sliding artefact that
  specular reflection causes on grazing incidence against a circular boundary.

> **Implementation:** `car_model.py → Car`

---

## Geometry & Physics

### Platform-to-Ground Geometry

The platform is at altitude `h` (ft) and slant range `R_s` (nm) from the cell
centre.  All internal calculations use nautical miles.

```
ground_range = sqrt(R_s^2 - h_nm^2)         where h_nm = h / 6076.12
depression_angle = arctan(h_nm / ground_range)
```

**Constraint:** `R_s > h_nm`; otherwise the geometry is physically invalid
(the platform would need to be directly above the target).

### FOV-to-Ground Footprint

The sensor's angular FOV is projected onto the ground plane.  Using the
small-angle approximation corrected for depression angle:

```
footprint_nm = R_s * fov_rad / cos(depression_angle)
```

Where `fov_rad = fov_deg * pi / 180`.  Larger depression angles (more
overhead viewing) produce smaller footprints.  Near-nadir (`depression -> 90 deg`)
the cosine is clamped to 0.01 to avoid singularities.

### Ground Sample Distance (GSD)

```
GSD_ft = footprint_ft / sensor_pixels
```

Where `footprint_ft = footprint_nm * 6076.12` and `sensor_pixels` defaults to
1024.

### Point-in-Footprint Test

The sensor footprint is modelled as a rotated rectangle.  The containment test
uses inverse rotation to transform the query point into the footprint's local
frame:

```
local_x = (px - cx) * cos(-rot) - (py - cy) * sin(-rot)
local_y = (px - cx) * sin(-rot) + (py - cy) * cos(-rot)
inside = |local_x| <= w/2  AND  |local_y| <= h/2
```

> **Implementation:** `geometry.py`

---

## Efficiency Metrics

Beyond basic detection rate and TTFD, the simulation computes composite
efficiency metrics that enable fair comparison across fundamentally different
sensor modalities.

### Detection Timeliness Index (DTI)

The DTI is the normalised area under the cumulative detection probability
curve:

```
DTI = (1 / T) * integral_0^T Pd_cumulative(t) dt
```

Where `T` is the simulation duration and `Pd_cumulative(t)` is the fraction of
trials that have achieved at least one detection by time `t`.  A sensor that
detects instantly in every trial has DTI = 1.0; a sensor that never detects has
DTI = 0.0.  DTI captures both the detection rate and how early detections
occur.

### Coverage Ratio

Each sensor tracks the total ground area it sweeps during the simulation:

```
Coverage_ratio = total_area_swept / cell_area
```

ISR sensors with wide FOV and fast scan rates achieve high coverage ratios (>>1
means the cell is covered multiple times).  NTISR sensors with narrow FOV
typically achieve lower coverage ratios.

### Search Efficiency Index (SEI)

The SEI measures how efficiently a sensor converts area coverage into
detections:

```
SEI = DTI / Coverage_ratio
```

A high SEI indicates the sensor is making good use of the area it scans.  A
low SEI indicates the sensor is sweeping large areas but achieving relatively
few or late detections.  This metric is particularly useful for comparing ISR
(high coverage, moderate detection rate) against NTISR FMV (low coverage, high
detection quality per look).

---

## Detection Probability Theory

### Cookie-Cutter Detection Model

The simulation uses a **cookie-cutter** (or "definite-range") detection model:
the sensor either sees the target (when it falls inside the footprint) or does
not.  There is no gradual degradation with off-axis angle or range — detection
opportunities occur only when the target is geometrically inside the sensor's
projected FOV rectangle.

This is a standard simplification in search theory, introduced by Koopman (1946,
1980) and widely used in operations research.  The cookie-cutter model is
appropriate when the sensor has a well-defined FOV boundary and the detection
probability drops sharply outside the FOV, as is the case for narrow-FOV EO/IR
targeting pods.

> **Ref:** Koopman, B.O. (1980). *Search and Screening: General Principles with
> Historical Applications*. Pergamon Press. (Originally published 1946 as OEG
> Report No. 56.)

### Cumulative Detection Probability

For sensors that dwell on a position (NTISR S&S, NTISR FMV), the detection
probability accumulates over time.  The model is calibrated so that one full
dwell period yields a cumulative detection probability equal to `pd_in_fov`.

The per-step detection probability is derived from the complementary survival
probability:

```
P(no detection in dwell T) = 1 - P_d
=> P(no detection per step dt) = (1 - P_d)^(dt / T_eff)
=> p_step = 1 - (1 - P_d)^(dt / T_eff)
```

Where:
- `P_d` = `pd_in_fov` (the configured base detection probability)
- `T_eff` = effective dwell time = `dwell_time_s - min_dwell_s` (for S&S)
  or `T_eff = 2.0 s` (for FMV, the recognition window)
- `dt` = simulation time step

This formulation treats each `dt`-second interval as an independent Bernoulli
trial, with the constraint that the cumulative probability over the full dwell
exactly equals `P_d`.  It follows the classical exponential detection model
from search theory.

> **Ref:** Stone, L.D. (1975). *Theory of Optimal Search*. Academic Press.

> **Ref:** Washburn, A.R. (2002). *Search and Detection*. 4th ed. INFORMS.

### Operator Recognition Model (FMV)

The FMV sensor adds a human-in-the-loop element.  The operator must:

1. **Recognise** the target — requiring continuous dwell of at least
   `min_dwell_s` (default 0.3 s).
2. **Confirm** the target — after recognition, Pd accumulates over a 2 s
   recognition window using the same exponential formula.

If the target leaves the FOV before recognition, the dwell counter resets.
This models the cognitive load of visual search: the operator needs a minimum
exposure time to distinguish a target from clutter.

The recognition time is consistent with the Johnson criteria for target
discrimination at tactical ranges.

> **Ref:** Johnson, J. (1958). "Analysis of image forming systems."
> *Proceedings of the Image Intensifier Symposium*, US Army Engineer R&D Labs,
> Fort Belvoir, VA, pp. 244–273.

> **Ref:** Holst, G.C. (2008). *Electro-Optical Imaging System Performance*.
> 5th ed. JCD Publishing / SPIE Press.

### Calibration Method

The detection probability calibration ensures physical consistency:

**ISR (per-frame model):** Each frame is an independent Bernoulli trial.  The
probability of detection in at least one frame over a full scan cycle of `N`
frames is:

```
P(detect in cycle) = 1 - (1 - p_frame)^N
```

Where `p_frame = pd_in_fov` and `N = len(scan_positions)`.  If the target
remains stationary in one FOV position for the entire cycle, the cumulative
probability increases exponentially.

**NTISR S&S (cumulative dwell model):** The per-step `p_step` is calibrated
such that:

```
P(detect in one dwell) = 1 - product_{k=1}^{K} (1 - p_step)
                        = 1 - (1 - p_step)^K
                        = pd_in_fov
```

Where `K = T_eff / dt` is the number of detection-eligible steps in one dwell.
Solving: `p_step = 1 - (1 - pd_in_fov)^(1/K) = 1 - (1 - pd_in_fov)^(dt/T_eff)`.

---

## Mathematical Formulas & Models

This section collects the principal equations implemented in the simulation,
with derivations and source references.

### Slant-Range Geometry

Given platform altitude `h` (ft) and slant range `R_s` (nm):

```
h_nm  = h / 6076.12                        [ft -> nm conversion]
R_g   = sqrt(R_s^2 - h_nm^2)               [ground range, nm]
delta = arctan(h_nm / R_g)                  [depression angle, rad]
```

The depression angle determines the projection geometry.  As `delta -> pi/2`
(nadir), the footprint becomes its minimum size.  As `delta -> 0` (grazing),
the footprint stretches toward infinity.

> **Ref:** Merrill, R.G. and Gunderson, D.E. (2000). *Airborne Reconnaissance
> Systems*. Chapter 4, "Sensor-Platform Geometry." ASPRS Manual of Remote
> Sensing, 3rd ed.

### FOV-to-Footprint Projection

```
w_ground = R_s * theta_fov / cos(delta)
```

Where `theta_fov = fov_deg * pi / 180`.  This is the standard small-angle
projection of angular FOV onto a slant plane, corrected by the depression-
angle cosine factor.

For a square FOV (as modelled), both width and height equal `w_ground`.
The footprint area is:

```
A_footprint = w_ground^2
```

> **Ref:** Holst, G.C. (2008). *Electro-Optical Imaging System Performance*.
> Chapter 3, "Optical Systems." SPIE Press.

### Boustrophedon Scan Pattern

The raster grid positions are computed as follows:

```
step = w_ground * 0.8                       [20% overlap factor]
N_x  = floor(2R_cell / step) + 1            [positions per row]
N_y  = floor(2R_cell / step) + 1            [number of rows]

For each row iy = 0, ..., N_y-1:
    For each column ix = 0, ..., N_x-1:
        x = cx - R_cell + ix * step
        y = cy - R_cell + iy * step
        IF (x - cx)^2 + (y - cy)^2 <= R_cell^2:
            include position (x, y)
    IF iy is odd:
        reverse the row                     [serpentine path]
```

The total number of raster positions `N_pos` determines the scan cycle time:

```
T_cycle = N_pos / frame_rate_hz             [seconds per full raster]
```

The 20% overlap factor is a design choice balancing coverage completeness
against scan speed.  It ensures that targets near the boundary between
adjacent footprints are covered by at least one frame.

> **Ref:** Choset, H. (2001). "Coverage of Known Spaces: The Boustrophedon
> Cellular Decomposition." *Autonomous Robots*, 9(3), pp. 247–253.

> **Ref:** Galceran, E. and Carreras, M. (2013). "A survey on coverage path
> planning for robotics." *Robotics and Autonomous Systems*, 61(12),
> pp. 1258–1276.

### Reflective Boundary Motion

The car moves at constant speed `v` with heading `psi`:

```
x_new = x + v * cos(psi) * dt
y_new = y + v * sin(psi) * dt
```

If the new position would be outside the cell (`||p_new - c|| > R_cell`):

```
n = (p_new - c) / ||p_new - c||            [outward unit normal]
inward_angle = arctan2(-n_y, -n_x)         [angle of inward normal]
jitter = U(-pi/6, pi/6)                    [±30° random offset]
psi_new = inward_angle + jitter            [new heading into the cell]
contact = c + n * R_cell                   [boundary contact point]
p_new = contact + [cos(psi_new), sin(psi_new)] * overshoot
```

The heading-change timer is reset to a fresh exponential draw so the car
maintains its inward heading long enough to move away from the edge.  A safety
clamp at 90% of the cell radius catches any residual edge cases.

This inward-cone approach replaces the classical specular reflection, which
can cause the car to slide along the boundary at grazing incidence angles.

> **Ref:** Chandrasekhar, S. (1943). "Stochastic Problems in Physics and
> Astronomy." *Reviews of Modern Physics*, 15(1), pp. 1–89.

### Per-Step Detection Probability

**ISR (independent frame model):**

```
p_detect = pd_in_fov    if target in footprint
         = 0            otherwise
```

Each frame is independent.  Over `K` frames with the target in the FOV:

```
P_cumulative = 1 - (1 - pd_in_fov)^K
```

**NTISR S&S (cumulative dwell model):**

After the minimum dwell threshold, each eligible time step contributes:

```
p_step = 1 - (1 - P_d)^(dt / T_eff)
```

Where `T_eff = dwell_time_s - min_dwell_s`.  Over a complete dwell, the
cumulative probability is:

```
P_dwell = 1 - (1 - p_step)^K = 1 - ((1 - P_d)^(dt/T_eff))^(T_eff/dt) = P_d
```

This self-consistency is exact regardless of the time step `dt`.

**NTISR FMV (recognition + accumulation):**

```
IF continuous_dwell >= min_dwell_s:
    p_step = 1 - (1 - P_d)^(dt / 2.0)
ELSE:
    p_step = 0
```

The 2.0 s recognition window means that 2 s of continuous post-recognition
dwell yields cumulative Pd = `pd_in_fov`.

### Monte Carlo Aggregation

After `N` trials, the aggregate statistics are:

```
detection_rate = (number of trials with at least one detection) / N

mean_TTFD  = mean(TTFD_i for all trials where TTFD < infinity)
median_TTFD = median(TTFD_i for all trials where TTFD < infinity)

cumulative_Pd(t) = mean over trials of indicator[detected by time t]
```

Confidence intervals on detection rate follow the binomial proportion:

```
SE = sqrt(p * (1-p) / N)
95% CI = p +/- 1.96 * SE
```

With the default 500 trials, a 50% detection rate has SE = 2.2%, yielding a
95% CI of +/- 4.4 percentage points.

> **Ref:** Robert, C.P. and Casella, G. (2004). *Monte Carlo Statistical
> Methods*. 2nd ed. Springer.

---

## Configuration Reference

All parameters are set in `config_default.yaml` and can be overridden via CLI
flags or the interactive menu.

### Platform Parameters

| Parameter | YAML Key | Default | Unit | Description |
|-----------|----------|---------|------|-------------|
| Altitude | `platform.altitude_ft` | 30000 | ft | Platform altitude AGL |
| Speed | `platform.speed_kts` | 420 | kts | Platform ground speed |
| Heading | `platform.heading_deg` | 0 | deg | Initial heading (0=North) |
| Slant Range | `geometry.slant_range_nm` | 10.0 | nm | LOS distance to cell centre |

### Cell & Target Parameters

| Parameter | YAML Key | Default | Unit | Description |
|-----------|----------|---------|------|-------------|
| Cell Radius | `cell.radius_nm` | 1.0 | nm | Circular search cell radius |
| Car Speed | `car.speed_kts` | 30 | kts | Target ground speed |
| Heading Change | `car.heading_change_interval_s` | 15 | s | Mean interval between heading changes |

### ISR Line Scanner Parameters

| Parameter | YAML Key | Default | Unit | Description |
|-----------|----------|---------|------|-------------|
| FOV | `isr.fov_deg` | 1.0 | deg | Angular width of one frame (**fixed hardware**) |
| Frame Rate | `isr.frame_rate_hz` | 30 | Hz | Frames per second (raster hop rate) |

### NTISR Step-and-Stare Parameters

| Parameter | YAML Key | Default | Unit | Description |
|-----------|----------|---------|------|-------------|
| FOV | `ntisr_step_stare.fov_deg` | 0.5 | deg | Narrow targeting pod FOV |
| Dwell Time | `ntisr_step_stare.dwell_time_s` | 2.0 | s | Time at each stare position |
| Slew Rate | `ntisr_step_stare.slew_rate_deg_s` | 20.0 | deg/s | Gimbal rotation speed |
| Scan Pattern | `ntisr_step_stare.scan_pattern` | boustrophedon | — | Raster pattern type |

### NTISR FMV Parameters

| Parameter | YAML Key | Default | Unit | Description |
|-----------|----------|---------|------|-------------|
| FOV | `ntisr_fmv.fov_deg` | 1.0 | deg | FMV search FOV |
| Slew Rate | `ntisr_fmv.slew_rate_deg_s` | 10.0 | deg/s | Operator panning speed |
| Speed Factor | `ntisr_fmv.search_speed_factor` | 0.5 | — | Fraction of max slew used for panning |
| Road Bias | `ntisr_fmv.road_bias` | 0.7 | 0–1 | Attraction toward road features |
| Revisit Tendency | `ntisr_fmv.revisit_tendency` | 0.3 | 0–1 | Tendency to revisit scanned areas |

### Detection Model Parameters

| Parameter | YAML Key | Default | Unit | Description |
|-----------|----------|---------|------|-------------|
| Pd in FOV | `detection.pd_in_fov` | 0.8 | 0–1 | Base detection probability per dwell |
| Decay Exponent | `detection.pd_decay_exponent` | 2.0 | — | Off-boresight Pd decay (reserved) |
| Min Dwell | `detection.min_dwell_for_detect_s` | 0.5 | s | Minimum dwell before detection eligible |

### Simulation Parameters

| Parameter | YAML Key | Default | Unit | Description |
|-----------|----------|---------|------|-------------|
| Duration | `simulation.duration_s` | 300 | s | Total simulation duration |
| Time Step | `simulation.time_step_s` | 0.1 | s | Simulation time step |
| MC Trials | `simulation.mc_trials` | 500 | — | Number of Monte Carlo trials |
| Random Seed | `simulation.random_seed` | 42 | — | Deterministic seed (null = random) |

### Advanced Parameters

| Parameter | YAML Key | Default | Unit | Description |
|-----------|----------|---------|------|-------------|
| N Targets | `advanced.n_targets` | 1 | — | Number of targets simultaneously in the cell |
| Parallel Workers | `advanced.parallel_workers` | 0 | — | MC worker processes (0 = auto, 1 = sequential) |
| Orbit Radius | `advanced.orbit_radius_nm` | 5.0 | nm | Racetrack orbit radius |
| Orbit Period | `advanced.orbit_period_s` | 300.0 | s | Time for one complete orbit |
| Orbit Cell Offset | `advanced.orbit_cell_offset_nm` | 8.0 | nm | Ground distance from orbit centre to cell |

---

## ISR Optimisation

The optimisation module (`optimize_isr_scan.py`) performs a systematic grid
search over ISR sensor and environmental parameters to find the combination
that maximises detection performance.

**Important:** The ISR FOV is a fixed hardware parameter and is **not** swept
in the optimisation.  It is taken from the base configuration (default 1.0 deg).

### Swept Parameters

| Parameter | Quick Grid | Full Grid |
|-----------|-----------|-----------|
| Frame Rate (Hz) | 10, 30, 80 | 5, 10, 20, 30, 50, 100 |
| Cell Radius (nm) | 0.5, 1.0, 1.5 | 0.3, 0.5, 1.0, 1.5, 2.0, 3.0 |
| Slant Range (nm) | 5.0, 10.0 | 5.0, 8.0, 10.0, 14.0, 20.0 |
| Altitude (ft) | 20000, 35000 | 15000, 20000, 25000, 30000, 35000, 45000 |

**Total combinations:** Quick = 3 x 3 x 2 x 2 = **36**; Full = 6 x 6 x 5 x 6 = **1,080**

For each combination, a reduced Monte Carlo simulation (default 50 trials) is
run and the detection rate, mean TTFD, and scan cycle time are recorded.

### Outputs

- **CSV file:** `results/isr_optimization.csv` — full results for all
  parameter combinations
- **Heatmap PDF:** `results/isr_optimization_heatmap.pdf` — detection rate
  colour maps (Frame Rate vs Cell Radius, panelled by Slant Range)
- **Console summary:** ranked table of top-N parameter sets

### Usage

```bash
python optimize_isr_scan.py                   # Full grid, 50 trials
python optimize_isr_scan.py --quick           # Quick grid for testing
python optimize_isr_scan.py --trials 100      # More trials per point
python optimize_isr_scan.py --top 20          # Show top-20 results
python optimize_isr_scan.py --no-heatmap      # Skip heatmap generation
```

---

## Advanced Analysis

The advanced analysis module (`advanced_analysis.py`) provides four additional
analyses that explore dimensions beyond the basic ISR-vs-NTISR comparison.
All analyses can be run standalone, from the interactive menu (option 15), or
embedded in the comprehensive report.

### Duration Crossover Analysis

Sweeps observation duration from 30 s to 600 s (configurable) to identify
the crossover point where slower sensor modes (NTISR) catch up to or surpass
faster ones (ISR) as more observation time becomes available.

**Outputs:**
- Detection rate vs duration (all three modes)
- Mean TTFD vs duration
- DTI and SEI vs duration
- CSV: `results/duration_crossover.csv`

**Default durations:** 30, 60, 120, 180, 300, 450, 600 seconds

### Multi-Platform Comparison

Compares detection performance across four predefined platform profiles, each
representing a different altitude and slant-range combination:

| Profile | Altitude | Slant Range |
|---------|----------|-------------|
| High-Alt ISR | 45,000 ft | 20 nm |
| Mid-Alt ISR | 30,000 ft | 10 nm |
| Low-Alt NTISR | 15,000 ft | 5 nm |
| Fighter Pod | 20,000 ft | 8 nm |

Each profile is run with all three sensor modes.  The analysis reveals which
sensor mode benefits most from closer range and which platform is most
efficient for each mission type.

**Outputs:**
- Detection rate grouped bar chart by platform
- SEI comparison by platform
- Summary table with all combinations
- CSV: `results/multi_platform.csv`

### Multi-Target Scenarios

Runs simulations with 1, 2, 3, 5, and 8 targets simultaneously in the cell.
Each target moves independently with its own random walk.  The analysis tracks:

- **Detect-any rate:** probability of finding at least one target
- **Mean fraction detected:** average percentage of all targets found

This reveals how each sensor mode scales with target density — wide-area ISR
benefits from more targets (higher chance of a target being in the scan path),
while NTISR FMV maintains high per-look detection quality.

**Outputs:**
- Detect-any rate vs target count
- Mean fraction detected vs target count
- CSV: `results/multi_target.csv`

### Moving Platform (Orbit) Analysis

Models a platform flying a circular racetrack orbit instead of hovering
stationary.  The orbit produces time-varying slant range as the platform
moves closer to and farther from the cell centre.

```
Platform position:  x(t) = R_orbit * cos(2*pi*t / T) + offset_x
                    y(t) = R_orbit * sin(2*pi*t / T)

Slant range:        SR(t) = sqrt(h^2 + x(t)^2 + y(t)^2)
```

The analysis samples the orbit at 6 equally-spaced phases and compares the
orbit-averaged detection rate against the stationary baseline.

**Outputs:**
- Detection rate: stationary vs orbit average (grouped bars)
- Detection rate along orbit phases (vs instantaneous slant range)

### Parallel MC Execution

Monte Carlo trials can be distributed across multiple CPU cores using Python's
`multiprocessing.Pool`.  This is transparent to the user — set
`advanced.parallel_workers` in the config (0 = use all available cores minus
one, 1 = sequential).

The parallel executor falls back gracefully to sequential execution if the
process pool fails to start.

### Per-Trial Raw Data Export

Exports every Monte Carlo trial's results to a CSV file for downstream
analysis in external tools (R, Excel, MATLAB, etc.).

```bash
# From the interactive menu: option 16
# Or programmatically:
python -c "
from mc_engine import run_mc
from advanced_analysis import export_per_trial_csv
import yaml
cfg = yaml.safe_load(open('config_default.yaml'))
results = {m: run_mc(cfg, m) for m in ['isr', 'ntisr_ss', 'ntisr_fmv']}
export_per_trial_csv(results, 'results/raw_data.csv')
"
```

**CSV columns:** `mode, trial, detected, ttfd_s, total_detections`

### Usage

```bash
# Run all advanced analyses (standalone)
python -c "
import yaml
from advanced_analysis import run_all_advanced_analyses
cfg = yaml.safe_load(open('config_default.yaml'))
run_all_advanced_analyses(cfg, trials_per_point=50, n_workers=4)
"

# From the interactive menu
python main.py --interactive
# Select option 15 for advanced analysis suite
# Select option 16 for per-trial CSV export
```

---

## Comprehensive Report

The report generator (`report_generator.py`) produces a multi-page PDF that
serves as both documentation and results archive:

1. **Methodology overview** — how the simulation works
2. **Sensor model descriptions** — each mode's physics and logic
3. **Target model** — car movement and boundary conditions
4. **Geometry** — footprint calculations and constraints
5. **Configuration summary** — all active parameter values
6. **ISR optimisation results** — parameter sweep table and heatmaps
7. **Sensitivity analysis** — one-at-a-time (OAT) parameter sweeps
8. **Advanced analyses** (optional) — duration crossover, multi-platform
   comparison, multi-target scaling, and moving platform orbit analysis

### Sensitivity Parameters Swept

| Parameter | Values |
|-----------|--------|
| ISR FOV | 0.2, 0.5, 1.0, 2.0, 3.0, 5.0 deg |
| ISR Frame Rate | 5, 10, 20, 30, 50, 100 Hz |
| Cell Radius | 0.3, 0.5, 1.0, 1.5, 2.0, 3.0 nm |
| Slant Range | 5.0, 8.0, 10.0, 14.0, 20.0 nm |
| Altitude | 15000, 20000, 25000, 30000, 35000 ft |
| Base Pd | 0.3, 0.5, 0.7, 0.8, 0.9, 1.0 |
| Car Speed | 10, 20, 30, 50, 80 kts |

Each produces a two-panel chart (Detection Rate and Mean TTFD vs parameter
value) and results are exported to `sensitivity_analysis.csv`.

### Usage

```bash
python report_generator.py                      # Default (quick optimisation)
python report_generator.py --full               # Full optimisation grid
python report_generator.py --no-optimization    # Skip optimisation sweep
python report_generator.py --opt-trials 100     # More trials per point
```

---

## Output Files

| File | Description |
|------|-------------|
| `results/ntisr_vs_isr_report.pdf` | Standard comparison report (from main.py) |
| `results/comprehensive_report.pdf` | Full report with manual, optimisation, sensitivity |
| `results/advanced_analysis.pdf` | Advanced analysis report (crossover, multi-platform, etc.) |
| `results/isr_optimization.csv` | ISR parameter sweep results |
| `results/isr_optimization_heatmap.pdf` | Optimisation heatmap visualisation |
| `results/sensitivity_analysis.csv` | OAT sensitivity sweep results |
| `results/duration_crossover.csv` | Detection rate vs observation duration |
| `results/multi_platform.csv` | Multi-platform comparison results |
| `results/multi_target.csv` | Multi-target scaling results |
| `results/per_trial_raw_data.csv` | Per-trial raw MC data (all modes) |
| `results/animation_frames/` | Animation frame images (if enabled) |

---

## References & Sources

The simulation's models, algorithms, and parameters draw on the following
foundational works in search theory, electro-optical systems, and operations
research:

1. **Koopman, B.O.** (1946/1980). *Search and Screening: General Principles
   with Historical Applications*. Pergamon Press (reprint of the 1946 OEG
   Report No. 56).
   — Foundational work on search theory; introduced the cookie-cutter detection
   model and random search framework used in this simulation.

2. **Stone, L.D.** (1975). *Theory of Optimal Search*. Academic Press.
   — Rigorous treatment of optimal search theory, including exponential
   detection models and cumulative detection probability derivations.

3. **Washburn, A.R.** (2002). *Search and Detection*. 4th ed. INFORMS
   (Institute for Operations Research and Management Sciences).
   — Practical treatment of search theory for military operations research,
   including sensor sweep models and Monte Carlo simulation methodology.

4. **Koopman, B.O.** (1956). "The Theory of Search, Part I: Kinematic Bases."
   *Operations Research*, 4(3), pp. 324–346.
   — Mathematical foundations for search by a moving observer with definite-
   range detection.

5. **Koopman, B.O.** (1957). "The Theory of Search, Part III: The Optimum
   Distribution of Searching Effort." *Operations Research*, 5(5), pp. 613–626.
   — Optimal allocation of search effort across cells.

6. **Johnson, J.** (1958). "Analysis of image forming systems." *Proceedings
   of the Image Intensifier Symposium*, US Army Engineer R&D Labs, Fort Belvoir,
   VA, pp. 244–273.
   — Introduced the Johnson criteria for target detection, recognition, and
   identification based on line-pair resolution; informs the minimum dwell /
   recognition time parameters.

7. **Holst, G.C.** (2008). *Electro-Optical Imaging System Performance*.
   5th ed. JCD Publishing / SPIE Press.
   — Comprehensive reference for EO/IR sensor performance modelling, including
   FOV-to-footprint projection, GSD calculations, and atmospheric effects.

8. **Driggers, R.G., Friedman, M.H., and Nichols, J.M.** (2012). *Introduction
   to Infrared and Electro-Optical Systems*. 2nd ed. Artech House.
   — IR/EO system fundamentals including sensor geometry, detection range
   equations, and search scan patterns.

9. **Galceran, E. and Carreras, M.** (2013). "A survey on coverage path
   planning for robotics." *Robotics and Autonomous Systems*, 61(12),
   pp. 1258–1276.
   — Survey of boustrophedon and other coverage path planning algorithms used
   for the raster scan pattern.

10. **Choset, H.** (2001). "Coverage of Known Spaces: The Boustrophedon
    Cellular Decomposition." *Autonomous Robots*, 9(3), pp. 247–253.
    — Theoretical basis for the boustrophedon coverage pattern.

11. **Chandrasekhar, S.** (1943). "Stochastic Problems in Physics and
    Astronomy." *Reviews of Modern Physics*, 15(1), pp. 1–89.
    — Classic treatment of random walks and diffusion with reflecting barriers,
    the basis for the car's reflective boundary model.

12. **Robert, C.P. and Casella, G.** (2004). *Monte Carlo Statistical Methods*.
    2nd ed. Springer.
    — Comprehensive reference for Monte Carlo simulation methodology, confidence
    interval estimation, and convergence properties.

13. **Law, A.M.** (2015). *Simulation Modeling and Analysis*. 5th ed. McGraw-Hill.
    — Standard textbook for discrete-event simulation, variance reduction,
    output analysis, and validation of MC models.

14. **Merrill, R.G. and Gunderson, D.E.** (2000). "Sensor-Platform Geometry."
    In *Manual of Remote Sensing*, 3rd ed. ASPRS (American Society for
    Photogrammetry and Remote Sensing).
    — Airborne sensor geometry, slant-range calculations, and depression angle
    effects on ground footprint.

15. **Szeliski, R.** (2010). *Computer Vision: Algorithms and Applications*.
    Springer.
    — Projective geometry and camera models underlying the FOV-to-ground
    projection.

16. **Wagner, D.H., Mylander, W.C., and Sanders, T.J.** (1999). *Naval
    Operations Analysis*. 3rd ed. Naval Institute Press.
    — Military operations research methods including airborne search, patrol
    models, and detection probability estimation.

17. **Mangel, M.** (1981). "Search for a Randomly Moving Object." *SIAM
    Journal on Applied Mathematics*, 40(2), pp. 327–338.
    — Optimal search for a diffusing target with Brownian motion, relevant to
    the random-walk car model.

18. **Dell, R.F., Eagle, J.N., Martins, G.H.A., and Santos, A.G.** (1996).
    "Using Multiple Searchers in Constrained-Path, Moving-Target Search
    Problems." *Naval Research Logistics*, 43(4), pp. 463–480.
    — Search models for moving targets in constrained regions, informing the
    cell-based search framework.

---

*This README was generated as part of the NTISR vs ISR Operational Analysis project.*
