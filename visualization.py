"""
Visualization module: side-by-side animation and PDF report generation.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from mc_engine import MCResult
from geometry import ground_footprint_at_cell


def generate_pdf_report(results: dict, cfg: dict, output_path: str):
    """
    Generate a multi-page PDF comparing MC results across sensor modes.

    Args:
        results: dict mapping mode_name -> MCResult
        cfg: configuration dict
        output_path: path to output PDF
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with PdfPages(output_path) as pdf:
        # --- Page 1: Title and configuration summary ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        title_text = "NTISR vs ISR Operational Analysis\nMonte Carlo Comparison Report"
        ax.text(0.5, 0.85, title_text, transform=ax.transAxes,
                fontsize=18, ha='center', va='top', fontweight='bold')

        config_lines = [
            f"Platform: {cfg['platform']['altitude_ft']} ft AGL, "
            f"{cfg['platform']['speed_kts']} kts",
            f"Slant range: {cfg['geometry']['slant_range_nm']} nm",
            f"Cell radius: {cfg['cell']['radius_nm']} nm",
            f"Car speed: {cfg['car']['speed_kts']} kts",
            f"Simulation: {cfg['simulation']['duration_s']}s, "
            f"dt={cfg['simulation']['time_step_s']}s, "
            f"{cfg['simulation']['mc_trials']} trials",
            "",
            "ISR line scanner:",
            f"  FOV: {cfg['isr']['fov_deg']}°, "
            f"Frame rate: {cfg['isr']['frame_rate_hz']} Hz",
            "",
            "NTISR Step-and-Stare:",
            f"  FOV: {cfg['ntisr_step_stare']['fov_deg']}°, "
            f"Dwell: {cfg['ntisr_step_stare']['dwell_time_s']}s, "
            f"Slew: {cfg['ntisr_step_stare']['slew_rate_deg_s']}°/s",
            "",
            "NTISR FMV:",
            f"  FOV: {cfg['ntisr_fmv']['fov_deg']}°, "
            f"Slew: {cfg['ntisr_fmv']['slew_rate_deg_s']}°/s, "
            f"Road bias: {cfg['ntisr_fmv']['road_bias']}",
        ]
        ax.text(0.1, 0.65, '\n'.join(config_lines), transform=ax.transAxes,
                fontsize=10, va='top', fontfamily='monospace')

        # Footprint sizes
        fp_lines = ["\nGround Footprint Sizes:"]
        for label, fov_key in [("ISR frame", ('isr', 'fov_deg')),
                                ("NTISR S&S", ('ntisr_step_stare', 'fov_deg')),
                                ("NTISR FMV", ('ntisr_fmv', 'fov_deg'))]:
            fp_nm = ground_footprint_at_cell(
                cfg['platform']['altitude_ft'],
                cfg['geometry']['slant_range_nm'],
                cfg[fov_key[0]][fov_key[1]])
            fp_lines.append(f"  {label}: {fp_nm:.4f} nm ({fp_nm*6076:.0f} ft)")
        ax.text(0.1, 0.18, '\n'.join(fp_lines), transform=ax.transAxes,
                fontsize=10, va='top', fontfamily='monospace')
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: Summary statistics table ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.95, "Detection Performance Summary",
                transform=ax.transAxes, fontsize=16, ha='center',
                fontweight='bold')

        col_labels = ['Metric', 'ISR\n(Back-Scan)', 'NTISR\n(Step&Stare)',
                      'NTISR\n(FMV)']
        mode_order = ['isr', 'ntisr_ss', 'ntisr_fmv']
        mode_names = {'isr': 'ISR', 'ntisr_ss': 'NTISR S&S',
                      'ntisr_fmv': 'NTISR FMV'}

        def fmt_time(v):
            return f"{v:.1f}s" if np.isfinite(v) else "Never"

        rows = []
        # Overall detection fraction
        row = ['Detection rate (%)']
        for m in mode_order:
            if m in results:
                row.append(f"{results[m].overall_detect_fraction*100:.1f}%")
            else:
                row.append('N/A')
        rows.append(row)

        # Mean TTFD
        row = ['Mean time to 1st detect']
        for m in mode_order:
            if m in results:
                row.append(fmt_time(results[m].mean_ttfd))
            else:
                row.append('N/A')
        rows.append(row)

        # Median TTFD
        row = ['Median time to 1st detect']
        for m in mode_order:
            if m in results:
                row.append(fmt_time(results[m].median_ttfd))
            else:
                row.append('N/A')
        rows.append(row)

        # Mean total detections
        row = ['Mean detection events']
        for m in mode_order:
            if m in results:
                row.append(f"{np.mean(results[m].total_detections):.1f}")
            else:
                row.append('N/A')
        rows.append(row)

        table = ax.table(cellText=rows, colLabels=col_labels,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 2.0)
        for (row_i, col_i), cell in table.get_celld().items():
            if row_i == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white', fontweight='bold')
            elif row_i % 2 == 0:
                cell.set_facecolor('#D6E4F0')
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 3: CDF of time to first detection ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        colors = {'isr': '#4472C4', 'ntisr_ss': '#ED7D31',
                  'ntisr_fmv': '#70AD47'}
        duration = cfg['simulation']['duration_s']
        dt = cfg['simulation']['time_step_s']

        for m in mode_order:
            if m not in results:
                continue
            r = results[m]
            # Build CDF from time_to_first_detect
            sorted_ttfd = np.sort(r.time_to_first_detect[
                np.isfinite(r.time_to_first_detect)])
            if len(sorted_ttfd) == 0:
                continue
            cdf_y = np.arange(1, len(sorted_ttfd) + 1) / r.n_trials
            ax.plot(sorted_ttfd, cdf_y, color=colors[m], linewidth=2,
                    label=f"{mode_names[m]}")

        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Cumulative Detection Probability', fontsize=12)
        ax.set_title('CDF of Time to First Detection', fontsize=14,
                     fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 1.05)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 4: Mean cumulative Pd over time ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        time_axis = np.arange(results[mode_order[0]].n_steps) * dt

        for m in mode_order:
            if m not in results:
                continue
            r = results[m]
            ax.plot(time_axis, r.mean_cumulative_pd, color=colors[m],
                    linewidth=2, label=f"{mode_names[m]}")

        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Fraction of Trials with Detection', fontsize=12)
        ax.set_title('Cumulative Detection Fraction Over Time', fontsize=14,
                     fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 1.05)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 5: Histogram of detection counts ---
        fig, axes = plt.subplots(1, 3, figsize=(11, 5))
        for idx, m in enumerate(mode_order):
            if m not in results:
                continue
            r = results[m]
            ax = axes[idx]
            max_det = max(int(np.max(r.total_detections)), 1)
            bins = np.arange(0, max_det + 2) - 0.5
            ax.hist(r.total_detections, bins=bins, color=colors[m],
                    edgecolor='white', alpha=0.85)
            ax.set_title(mode_names[m], fontsize=12, fontweight='bold')
            ax.set_xlabel('Detection Events')
            ax.set_ylabel('Trials')
            ax.grid(True, alpha=0.3)
        fig.suptitle('Distribution of Detection Events per Trial',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF report saved to: {output_path}")


def generate_animation_frames(cfg: dict, output_dir: str, n_frames: int = 200):
    """
    Generate a single-trial animation showing all three sensor modes
    side by side with the car moving in the cell.

    Saves frames as PNG files for later assembly or interactive display.
    """
    os.makedirs(output_dir, exist_ok=True)

    from mc_engine import build_sensor
    from car_model import Car

    dt = cfg['simulation']['time_step_s']
    duration = min(cfg['simulation']['duration_s'], 60.0)  # Cap animation at 60s
    speed_factor = cfg['output'].get('animation_speed', 5.0)
    frame_dt = duration / n_frames
    steps_per_frame = max(1, int(frame_dt / dt))

    cell_cx, cell_cy = 0.0, 0.0
    cell_r = cfg['cell']['radius_nm']
    seed = cfg['simulation'].get('random_seed', 42)

    modes = ['isr', 'ntisr_ss', 'ntisr_fmv']
    titles = ['ISR (Back-Scan Mirror)', 'NTISR (Step & Stare)', 'NTISR (FMV Search)']
    colors = {'isr': '#4472C4', 'ntisr_ss': '#ED7D31', 'ntisr_fmv': '#70AD47'}

    # Create shared car (same trajectory for all modes in animation)
    rng_car = np.random.default_rng(seed)
    car = Car(cell_cx, cell_cy, cell_r,
              cfg['car']['speed_kts'],
              cfg['car']['heading_change_interval_s'], rng_car)

    # Pre-compute car trajectory
    total_steps = int(duration / dt)
    car_positions = np.zeros((total_steps, 2))
    for s in range(total_steps):
        car.step(dt)
        car_positions[s] = car.get_position()

    # Create sensors
    sensors = {}
    for m in modes:
        rng_s = np.random.default_rng(seed + hash(m) % 10000)
        sensors[m] = build_sensor(m, cfg, cell_cx, cell_cy, rng_s)

    # Generate frames
    print(f"Generating {n_frames} animation frames...")
    for frame_i in range(n_frames):
        step_start = frame_i * steps_per_frame
        step_end = min(step_start + steps_per_frame, total_steps)
        if step_start >= total_steps:
            break

        t = step_end * dt

        # Advance sensors to current time
        for s_idx in range(step_start, step_end):
            for m in modes:
                sensors[m].step(s_idx * dt, dt)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        car_x, car_y = car_positions[min(step_end - 1, total_steps - 1)]

        for idx, m in enumerate(modes):
            ax = axes[idx]
            # Cell boundary
            circle = plt.Circle((cell_cx, cell_cy), cell_r,
                                fill=False, color='gray', linewidth=2)
            ax.add_patch(circle)

            # Sensor footprint
            fp = sensors[m].get_footprint()
            if fp['w'] > 0 and fp['h'] > 0:
                rect = patches.Rectangle(
                    (fp['x'] - fp['w']/2, fp['y'] - fp['h']/2),
                    fp['w'], fp['h'],
                    angle=np.degrees(fp['rot']),
                    linewidth=2, edgecolor=colors[m],
                    facecolor=colors[m], alpha=0.3
                )
                ax.add_patch(rect)

            # Car position
            ax.plot(car_x, car_y, 'ro', markersize=8, zorder=5)

            # Car trail
            trail_start = max(0, step_end - 50)
            trail = car_positions[trail_start:step_end]
            if len(trail) > 1:
                ax.plot(trail[:, 0], trail[:, 1], 'r-', alpha=0.3, linewidth=1)

            # FMV roads (if applicable)
            if m == 'ntisr_fmv' and hasattr(sensors[m], 'roads'):
                for (rx, ry, ra) in sensors[m].roads:
                    dx = cell_r * np.cos(ra)
                    dy = cell_r * np.sin(ra)
                    ax.plot([rx - dx, rx + dx], [ry - dy, ry + dy],
                            'k--', alpha=0.2, linewidth=1)

            ax.set_xlim(-cell_r * 1.2, cell_r * 1.2)
            ax.set_ylim(-cell_r * 1.2, cell_r * 1.2)
            ax.set_aspect('equal')
            ax.set_title(f"{titles[idx]}\nt = {t:.1f}s", fontsize=11)
            ax.grid(True, alpha=0.2)

        fig.suptitle('NTISR vs ISR — Sensor Scan Comparison',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"frame_{frame_i:04d}.png"),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)

        if (frame_i + 1) % 50 == 0:
            print(f"  Frame {frame_i + 1}/{n_frames}")

    print(f"Animation frames saved to: {output_dir}")
