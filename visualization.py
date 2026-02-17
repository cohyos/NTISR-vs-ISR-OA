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

    Only includes the sensors that were actually run (present in results).

    Args:
        results: dict mapping mode_key -> MCResult (only active modes)
        cfg: configuration dict
        output_path: path to output PDF
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    all_mode_names = {
        'isr': 'ISR (Back-Scan)',
        'ntisr_ss': 'NTISR (Step & Stare)',
        'ntisr_fmv': 'NTISR (FMV)',
    }
    all_colors = {'isr': '#4472C4', 'ntisr_ss': '#ED7D31',
                  'ntisr_fmv': '#70AD47'}

    # Only include modes that were actually simulated
    active_modes = [m for m in ['isr', 'ntisr_ss', 'ntisr_fmv']
                    if m in results]
    n_modes = len(active_modes)
    if n_modes == 0:
        print("No results to report.")
        return

    duration = cfg['simulation']['duration_s']
    dt = cfg['simulation']['time_step_s']

    # Sensor config blocks keyed by mode
    sensor_config_lines = {
        'isr': [
            "ISR line scanner:",
            f"  FOV: {cfg['isr']['fov_deg']}deg, "
            f"Frame rate: {cfg['isr']['frame_rate_hz']} Hz",
        ],
        'ntisr_ss': [
            "NTISR Step-and-Stare:",
            f"  FOV: {cfg['ntisr_step_stare']['fov_deg']}deg, "
            f"Dwell: {cfg['ntisr_step_stare']['dwell_time_s']}s, "
            f"Slew: {cfg['ntisr_step_stare']['slew_rate_deg_s']}deg/s",
        ],
        'ntisr_fmv': [
            "NTISR FMV:",
            f"  FOV: {cfg['ntisr_fmv']['fov_deg']}deg, "
            f"Slew: {cfg['ntisr_fmv']['slew_rate_deg_s']}deg/s, "
            f"Road bias: {cfg['ntisr_fmv']['road_bias']}",
        ],
    }
    fov_keys = {
        'isr': ('isr', 'fov_deg', 'ISR frame'),
        'ntisr_ss': ('ntisr_step_stare', 'fov_deg', 'NTISR S&S'),
        'ntisr_fmv': ('ntisr_fmv', 'fov_deg', 'NTISR FMV'),
    }

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
            f"Simulation: {duration}s, dt={dt}s, "
            f"{cfg['simulation']['mc_trials']} trials",
        ]
        for m in active_modes:
            config_lines.append("")
            config_lines.extend(sensor_config_lines[m])

        ax.text(0.1, 0.65, '\n'.join(config_lines), transform=ax.transAxes,
                fontsize=10, va='top', fontfamily='monospace')

        # Footprint sizes — only for active sensors
        fp_lines = ["\nGround Footprint Sizes:"]
        for m in active_modes:
            sec, key, label = fov_keys[m]
            fp_nm = ground_footprint_at_cell(
                cfg['platform']['altitude_ft'],
                cfg['geometry']['slant_range_nm'],
                cfg[sec][key])
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

        col_labels = ['Metric'] + [all_mode_names[m] for m in active_modes]

        def fmt_time(v):
            return f"{v:.1f}s" if np.isfinite(v) else "Never"

        rows = [
            ['Detection rate (%)'] + [
                f"{results[m].overall_detect_fraction*100:.1f}%"
                for m in active_modes],
            ['Mean time to 1st detect'] + [
                fmt_time(results[m].mean_ttfd) for m in active_modes],
            ['Median time to 1st detect'] + [
                fmt_time(results[m].median_ttfd) for m in active_modes],
            ['Mean detection events'] + [
                f"{np.mean(results[m].total_detections):.1f}"
                for m in active_modes],
        ]

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

        for m in active_modes:
            r = results[m]
            sorted_ttfd = np.sort(r.time_to_first_detect[
                np.isfinite(r.time_to_first_detect)])
            if len(sorted_ttfd) == 0:
                continue
            cdf_y = np.arange(1, len(sorted_ttfd) + 1) / r.n_trials
            ax.step(sorted_ttfd, cdf_y, where='post',
                    color=all_colors[m], linewidth=2,
                    label=f"{all_mode_names[m]} "
                          f"({r.overall_detect_fraction*100:.0f}% final)")

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

        # --- Page 4: Detection rate by time window ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        n_bins = min(20, max(5, int(duration / 15)))
        bin_edges = np.linspace(0, duration, n_bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        bar_width = bin_width / (n_modes + 1)

        for i, m in enumerate(active_modes):
            r = results[m]
            finite_ttfd = r.time_to_first_detect[
                np.isfinite(r.time_to_first_detect)]
            counts, _ = np.histogram(finite_ttfd, bins=bin_edges)
            rate = counts / r.n_trials * 100  # percent of trials
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            offset = (i - (n_modes - 1) / 2) * bar_width
            ax.bar(bin_centers + offset, rate, width=bar_width * 0.9,
                   color=all_colors[m], alpha=0.85,
                   label=all_mode_names[m])

        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('New Detections (% of trials)', fontsize=12)
        ax.set_title('First-Detection Rate by Time Window', fontsize=14,
                     fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(0, duration)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 5: Histogram of detection counts ---
        fig, axes = plt.subplots(1, n_modes, figsize=(11, 5), squeeze=False)
        for idx, m in enumerate(active_modes):
            r = results[m]
            ax = axes[0, idx]
            max_det = max(int(np.max(r.total_detections)), 1)
            bins = np.arange(0, max_det + 2) - 0.5
            ax.hist(r.total_detections, bins=bins, color=all_colors[m],
                    edgecolor='white', alpha=0.85)
            ax.set_title(all_mode_names[m], fontsize=12, fontweight='bold')
            ax.set_xlabel('Detection Events')
            ax.set_ylabel('Trials')
            ax.grid(True, alpha=0.3)
        fig.suptitle('Distribution of Detection Events per Trial',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF report saved to: {output_path}")


def generate_animation_frames(cfg: dict, output_dir: str,
                               active_modes: list = None):
    """
    Generate a single-trial animation showing active sensor modes
    side by side with the car moving in the cell.

    Saves frames as PNG files, then assembles them into a playable MP4 video
    with a timestamp in the filename.  Frame count scales automatically with
    simulation duration (~2 frames per second of simulated time, capped at 600).

    Args:
        cfg: configuration dict
        output_dir: directory to save frames and video
        active_modes: list of mode keys to animate (e.g. ['isr', 'ntisr_ss']).
                      If None, defaults to all three modes.
    """
    os.makedirs(output_dir, exist_ok=True)

    from mc_engine import build_sensor
    from car_model import Car

    dt = cfg['simulation']['time_step_s']
    duration = cfg['simulation']['duration_s']
    speed_factor = cfg['output'].get('animation_speed', 5.0)

    # Scale frame count with duration: ~2 frames per simulated second, capped
    n_frames = min(600, max(20, int(duration * 2)))

    frame_dt = duration / n_frames
    steps_per_frame = max(1, int(frame_dt / dt))

    cell_cx, cell_cy = 0.0, 0.0
    cell_r = cfg['cell']['radius_nm']
    seed = cfg['simulation'].get('random_seed', 42)

    all_titles = {
        'isr': 'ISR (Back-Scan Mirror)',
        'ntisr_ss': 'NTISR (Step & Stare)',
        'ntisr_fmv': 'NTISR (FMV Search)',
    }
    all_colors = {'isr': '#4472C4', 'ntisr_ss': '#ED7D31', 'ntisr_fmv': '#70AD47'}

    # Determine active modes
    if active_modes is None:
        modes = ['isr', 'ntisr_ss', 'ntisr_fmv']
    else:
        modes = [m for m in active_modes if m in all_titles]
    n_modes = len(modes)
    if n_modes == 0:
        print("No active sensor modes — skipping animation.")
        return

    # Create shared car with a unique seed so the animation doesn't
    # always replay the exact same MC trial (trial 0 with base seed).
    # Mixing in the current time makes each run visually distinct.
    import time as _time
    anim_seed = (seed if seed is not None else 0) + int(_time.time()) % 100000
    rng_car = np.random.default_rng(anim_seed)
    car = Car(cell_cx, cell_cy, cell_r,
              cfg['car']['speed_kts'],
              cfg['car']['heading_change_interval_s'], rng_car)

    # Pre-compute car trajectory
    total_steps = int(duration / dt)
    car_positions = np.zeros((total_steps, 2))
    for s in range(total_steps):
        car.step(dt)
        car_positions[s] = car.get_position()

    # Create sensors only for active modes
    sensors = {}
    for m in modes:
        rng_s = np.random.default_rng(anim_seed + hash(m) % 10000)
        sensors[m] = build_sensor(m, cfg, cell_cx, cell_cy, rng_s)

    # Adaptive figure sizing: 6 inches per panel
    fig_width = max(6, 6 * n_modes)
    fig_height = 6

    # Generate frames
    print(f"Generating {n_frames} animation frames for {n_modes} sensor(s)...")
    frame_paths = []
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

        fig, axes = plt.subplots(1, n_modes, figsize=(fig_width, fig_height),
                                 squeeze=False)
        car_x, car_y = car_positions[min(step_end - 1, total_steps - 1)]

        for idx, m in enumerate(modes):
            ax = axes[0, idx]
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
                    linewidth=2, edgecolor=all_colors[m],
                    facecolor=all_colors[m], alpha=0.3
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
            ax.set_title(f"{all_titles[m]}\nt = {t:.1f}s", fontsize=11)
            ax.grid(True, alpha=0.2)

        fig.suptitle('NTISR vs ISR — Sensor Scan Comparison',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        frame_path = os.path.join(output_dir, f"frame_{frame_i:04d}.png")
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(frame_path)

        if (frame_i + 1) % 50 == 0:
            print(f"  Frame {frame_i + 1}/{n_frames}")

    print(f"Animation frames saved to: {output_dir}")

    # Assemble frames into a playable MP4 video
    if frame_paths:
        _assemble_video(frame_paths, output_dir, fps=20)


def _pad_even(img):
    """Pad image to even width and height (required by libx264)."""
    h, w = img.shape[:2]
    new_h = h if h % 2 == 0 else h + 1
    new_w = w if w % 2 == 0 else w + 1
    if new_h != h or new_w != w:
        padded = np.full((new_h, new_w) + img.shape[2:], 255, dtype=img.dtype)
        padded[:h, :w] = img
        return padded
    return img


def _assemble_video(frame_paths: list, output_dir: str, fps: int = 20):
    """
    Assemble PNG frames into a playable MP4 video with a timestamp in the
    filename. Saves the video in the parent of the frames directory (i.e. the
    main output directory).

    Tries three approaches in order:
      1. imageio-ffmpeg plugin (best quality, requires pip install imageio-ffmpeg)
      2. Pillow animated GIF fallback (always available, larger file)
      3. Gives up gracefully — frames are still available as PNGs
    """
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = os.path.dirname(output_dir) or output_dir

    print(f"Assembling {len(frame_paths)} frames into video at {fps} fps...")

    # --- Approach 1: imageio + ffmpeg → MP4 ---
    try:
        import imageio
        import imageio_ffmpeg  # noqa: F401  — ensure ffmpeg backend exists
        video_path = os.path.join(video_dir, f"sensor_animation_{timestamp}.mp4")
        writer = imageio.get_writer(video_path, format='FFMPEG', fps=fps,
                                    codec='libx264', quality=8,
                                    macro_block_size=1)
        for p in frame_paths:
            writer.append_data(_pad_even(imageio.imread(p)))
        writer.close()
        print(f"Video saved to: {video_path}")
        return
    except ImportError:
        print("  imageio-ffmpeg not installed — trying Pillow GIF fallback...")
    except Exception as e:
        print(f"  FFmpeg approach failed ({e}) — trying Pillow GIF fallback...")

    # --- Approach 2: Pillow animated GIF ---
    try:
        from PIL import Image
        gif_path = os.path.join(video_dir, f"sensor_animation_{timestamp}.gif")
        imgs = [Image.open(p) for p in frame_paths]
        frame_duration_ms = int(1000 / fps)
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                     duration=frame_duration_ms, loop=0)
        print(f"Animated GIF saved to: {gif_path}")
        return
    except Exception as e2:
        print(f"  GIF fallback also failed ({e2}).")

    print("  Frames are still available as individual PNGs.")
