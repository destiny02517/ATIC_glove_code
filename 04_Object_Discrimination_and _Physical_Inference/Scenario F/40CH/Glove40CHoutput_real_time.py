import csv
import os
import re
import time
from collections import deque
from typing import Dict, List, Tuple

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import ttk

try:
    import serial
    from serial.tools import list_ports
except ImportError as exc:
    raise ImportError(
        "pyserial is required. Install it with: pip install pyserial"
    ) from exc


# ===================== Config =====================
LAYOUT_IMG = "Layout5_00.jpg"
PORT = "COM5"
BAUDRATE = 2000000

SERIAL_TIMEOUT = 0.0

CHANNEL_COUNT = 40
SAMPLE_RATE_HZ = 500
DISPLAY_SECONDS = 6
HISTORY_LEN = SAMPLE_RATE_HZ * DISPLAY_SECONDS
REFRESH_MS = 20
STACKED_SPACING = 0.035
WAVE_GAIN = 1.0
AUTO_BASELINE_SECONDS = 1.0
HEATMAP_DEADBAND = 0.02
DELTA_POSITIVE_ONLY = True
MAX_RENDER_FRAMES_PER_UPDATE = 80
HEATMAP_ATTACK_ALPHA = 0.85
HEATMAP_RELEASE_ALPHA = 0.20
HEATMAP_ZERO_HOLD_THRESHOLD = 0.01
HEATMAP_RELEASE_TO_ZERO_THRESHOLD = 0.04

HEATMAP_MODE = "delta"   # "raw" or "delta"
USE_AUTO_RANGE = True
GLOBAL_MIN = 0.0
GLOBAL_MAX = 4095.0

LIGHT_HEX = "#eee8e5"
DARK_HEX = "#44352e"
SAVE_DIR = "realtime_output"
DRAW_HEATMAP_NUMBERS_LIVE = False
HEATMAP_USE_HISTORY_RANGE = False
HEATMAP_MIN_SPAN = 0.05
HEATMAP_RANGE_SMOOTHING = 0.85

NEW_NUMBERS = {
    0: 38, 1: 5, 2: 31, 3: 3, 4: 30, 5: 25, 6: 17, 7: 2, 8: 24, 9: 26,
    10: 29, 11: 15, 12: 14, 13: 27, 14: 34, 15: 4, 16: 28, 17: 7, 18: 10, 19: 0,
    20: 35, 21: 6, 22: 21, 23: 1, 24: 13, 25: 37, 26: 9, 27: 20, 28: 11, 29: 32,
    30: 18, 31: 12, 32: 36, 33: 8, 34: 23, 35: 33, 36: 19, 37: 16, 38: 39, 39: 22,
}


# ===================== Helpers =====================
def hex_to_rgb01(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    rgb = [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]
    return np.array(rgb, dtype=float) / 255.0


def make_colormap(light_hex: str, dark_hex: str):
    return mcolors.LinearSegmentedColormap.from_list(
        "custom", [hex_to_rgb01(light_hex), hex_to_rgb01(dark_hex)], N=256
    )


def detect_red_contours(image_bgr: np.ndarray):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 200]
    return contours, mask


def contour_centroid(contour) -> Tuple[int, int]:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return 0, 0
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def render_heatmap_image(
    layout_img_bgr: np.ndarray,
    contours,
    contour_channels: List[int],
    contour_centers: List[Tuple[int, int]],
    values: np.ndarray,
    cmap,
    value_min: float,
    value_max: float,
    draw_numbers: bool = True,
) -> np.ndarray:
    img = layout_img_bgr.copy()
    values = np.asarray(values, dtype=float)

    span = max(value_max - value_min, 1e-12)
    norm = np.clip((values - value_min) / span, 0.0, 1.0)

    for contour_idx, contour in enumerate(contours):
        channel_idx = contour_channels[contour_idx]
        if channel_idx is None or not (0 <= channel_idx < len(values)):
            continue
        rgba = cmap(norm[channel_idx])
        bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
        cv2.drawContours(img, [contour], -1, bgr, -1)

    if not draw_numbers:
        return img

    for contour_idx, _contour in enumerate(contours):
        channel_idx = contour_channels[contour_idx]
        if channel_idx is None or not (0 <= channel_idx < len(values)):
            continue
        value = values[channel_idx]
        cx, cy = contour_centers[contour_idx]
        cv2.putText(
            img,
            f"{value:.1f}",
            (cx - 14, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return img


def parse_serial_line(line: str, expected_channels: int = CHANNEL_COUNT):
    parts = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
    if len(parts) < expected_channels:
        return None
    try:
        values = np.array([float(x) for x in parts[:expected_channels]], dtype=float)
    except ValueError:
        return None
    return values


def list_available_ports() -> List[str]:
    return [port.device for port in list_ports.comports()]


# ===================== App =====================
class RealTimeGloveApp:
    def __init__(self):
        self.layout_img_bgr = cv2.imread(LAYOUT_IMG)
        if self.layout_img_bgr is None:
            raise FileNotFoundError(f"Cannot read layout image: {LAYOUT_IMG}")

        self.contours, _ = detect_red_contours(self.layout_img_bgr)
        if not self.contours:
            raise RuntimeError("No red sensor contours were found in the layout image.")

        self.cmap = make_colormap(LIGHT_HEX, DARK_HEX)
        self.serial_port = None
        self.connected = False
        self.paused = False
        self.selected_port = PORT
        self.contour_channels = [NEW_NUMBERS.get(i) for i in range(len(self.contours))]
        self.contour_centers = [contour_centroid(contour) for contour in self.contours]

        self.latest_values = np.zeros(CHANNEL_COUNT, dtype=float)
        self.baseline = None
        self.sample_count = 0
        self.last_frame_time = 0.0
        self.start_time = None
        self.auto_baseline_sum = np.zeros(CHANNEL_COUNT, dtype=float)
        self.auto_baseline_count = 0
        self.auto_baseline_done = False
        self.serial_text_buffer = ""
        self.heatmap_value_min = 0.0
        self.heatmap_value_max = 1.0
        self.display_heatmap_values = np.zeros(CHANNEL_COUNT, dtype=float)

        self.history = deque(maxlen=HISTORY_LEN)
        for _ in range(HISTORY_LEN):
            self.history.append(np.zeros(CHANNEL_COUNT, dtype=float))
        self.saved_samples = []

        os.makedirs(SAVE_DIR, exist_ok=True)

        self._build_figure()

    def _build_figure(self):
        self.fig = plt.figure(figsize=(15, 8))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0])
        self.ax_sig = self.fig.add_subplot(gs[0, 0])
        self.ax_map = self.fig.add_subplot(gs[0, 1])
        self.ax_map.axis("off")
        self.ax_sig.set_facecolor("#f4f4f4")

        x = np.linspace(-DISPLAY_SECONDS, 0.0, HISTORY_LEN)
        colors = plt.cm.tab20(np.linspace(0, 1, CHANNEL_COUNT))
        self.lines = []
        self.channel_offsets = np.arange(CHANNEL_COUNT, dtype=float) * STACKED_SPACING
        for idx in range(CHANNEL_COUNT):
            y0 = np.full(HISTORY_LEN, self.channel_offsets[idx], dtype=float)
            line, = self.ax_sig.plot(x, y0, linewidth=0.8, color=colors[idx % len(colors)], alpha=0.95)
            self.lines.append(line)

        self.ax_sig.set_xlim(-DISPLAY_SECONDS, 0.0)
        self.ax_sig.set_xlabel("Time (s)")
        self.ax_sig.set_ylabel("Relative level")
        self.ax_sig.set_title("40-Channel Live View")
        self.ax_sig.grid(True, linestyle="--", linewidth=0.7, color="#9e9e9e", alpha=0.6)
        self.ax_sig.yaxis.tick_right()
        self.ax_sig.yaxis.set_label_position("right")
        self.ax_sig.set_yticks(self.channel_offsets[::2])
        self.ax_sig.set_yticklabels([f"I{i}" for i in range(0, CHANNEL_COUNT, 2)], fontsize=8)

        initial_map = render_heatmap_image(
            self.layout_img_bgr,
            self.contours,
            self.contour_channels,
            self.contour_centers,
            self.latest_values,
            self.cmap,
            GLOBAL_MIN,
            GLOBAL_MAX,
            draw_numbers=DRAW_HEATMAP_NUMBERS_LIVE,
        )
        self.im_artist = self.ax_map.imshow(cv2.cvtColor(initial_map, cv2.COLOR_BGR2RGB))

        self.status_text = self.fig.text(
            0.02,
            0.97,
            self._build_status_text("Disconnected"),
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", alpha=0.2),
        )

        self._add_buttons()
        self.fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        self.timer = self.fig.canvas.new_timer(interval=REFRESH_MS)
        self.timer.add_callback(self.on_timer)
        self.timer.start()

    def _add_buttons(self):
        btn_y = 0.015
        btn_w = 0.12
        btn_h = 0.05
        gap = 0.015
        x0 = 0.06

        ax_connect = self.fig.add_axes([x0, btn_y, btn_w, btn_h])
        ax_pause = self.fig.add_axes([x0 + (btn_w + gap), btn_y, btn_w, btn_h])
        ax_base = self.fig.add_axes([x0 + 2 * (btn_w + gap), btn_y, btn_w, btn_h])
        ax_clear = self.fig.add_axes([x0 + 3 * (btn_w + gap), btn_y, btn_w, btn_h])
        ax_ports = self.fig.add_axes([x0 + 4 * (btn_w + gap), btn_y, btn_w, btn_h])
        ax_csv = self.fig.add_axes([x0 + 5 * (btn_w + gap), btn_y, btn_w, btn_h])
        ax_shot = self.fig.add_axes([x0 + 6 * (btn_w + gap), btn_y, btn_w, btn_h])

        self.btn_connect = Button(ax_connect, "Connect")
        self.btn_pause = Button(ax_pause, "Pause")
        self.btn_base = Button(ax_base, "Set Baseline")
        self.btn_clear = Button(ax_clear, "Clear Baseline")
        self.btn_ports = Button(ax_ports, "Select Port")
        self.btn_csv = Button(ax_csv, "Save CSV")
        self.btn_shot = Button(ax_shot, "Save Heatmap")

        self.btn_connect.on_clicked(self.toggle_connection)
        self.btn_pause.on_clicked(self.toggle_pause)
        self.btn_base.on_clicked(self.set_baseline)
        self.btn_clear.on_clicked(self.clear_baseline)
        self.btn_ports.on_clicked(self.select_port)
        self.btn_csv.on_clicked(self.save_csv)
        self.btn_shot.on_clicked(self.save_heatmap)

    def _build_status_text(self, state: str) -> str:
        ports = ", ".join(list_available_ports()) or "No COM ports"
        baseline_state = "ON" if self.baseline is not None else "OFF"
        mode = HEATMAP_MODE if self.baseline is not None else "raw"
        auto_base = "done" if self.auto_baseline_done else f"building {self.auto_baseline_count}/{int(AUTO_BASELINE_SECONDS * SAMPLE_RATE_HZ)}"
        return (
            f"State: {state} | Port: {self.selected_port} @ {BAUDRATE} | Samples: {self.sample_count}\n"
            f"Baseline: {baseline_state} | Auto baseline: {auto_base} | Heatmap mode: {mode} | Ports: {ports}"
        )

    def _set_status(self, state: str):
        self.status_text.set_text(self._build_status_text(state))
        self.fig.canvas.draw_idle()

    def toggle_connection(self, _event):
        if self.connected:
            self.disconnect_serial()
        else:
            self.connect_serial()

    def connect_serial(self):
        try:
            self.serial_port = serial.Serial(self.selected_port, BAUDRATE, timeout=SERIAL_TIMEOUT)
        except Exception as exc:
            self._set_status(f"Connect failed: {exc}")
            return

        self.connected = True
        self.serial_text_buffer = ""
        try:
            self.serial_port.reset_input_buffer()
        except Exception:
            pass
        if self.start_time is None:
            self.start_time = time.time()
        self.btn_connect.label.set_text("Disconnect")
        self._set_status("Connected")

    def disconnect_serial(self):
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
        self.serial_port = None
        self.connected = False
        self.btn_connect.label.set_text("Connect")
        self._set_status("Disconnected")

    def toggle_pause(self, _event):
        self.paused = not self.paused
        self.btn_pause.label.set_text("Resume" if self.paused else "Pause")
        self._set_status("Paused" if self.paused else ("Connected" if self.connected else "Disconnected"))

    def set_baseline(self, _event):
        self.baseline = self.latest_values.copy()
        self.auto_baseline_done = True
        self._set_status("Baseline captured")

    def clear_baseline(self, _event):
        self.baseline = None
        self.auto_baseline_sum.fill(0.0)
        self.auto_baseline_count = 0
        self.auto_baseline_done = False
        self._set_status("Baseline cleared")

    def select_port(self, _event):
        if self.connected:
            self._set_status("Disconnect first before changing port")
            return

        ports = list_available_ports()
        if not ports:
            self._set_status("No COM ports found")
            return

        selected = self._show_port_dialog(ports, self.selected_port)
        if selected:
            self.selected_port = selected
            self._set_status(f"Port selected: {selected}")

    def _show_port_dialog(self, ports: List[str], current_port: str):
        root = tk.Tk()
        root.withdraw()

        dialog = tk.Toplevel(root)
        dialog.title("Select Serial Port")
        dialog.resizable(False, False)
        dialog.grab_set()

        tk.Label(dialog, text="Serial Port").grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")
        var = tk.StringVar(value=current_port if current_port in ports else ports[0])
        combo = ttk.Combobox(dialog, textvariable=var, values=ports, state="readonly", width=18)
        combo.grid(row=1, column=0, padx=12, pady=6)
        combo.focus_set()

        result = {"value": None}

        def on_ok():
            result["value"] = var.get()
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        btn_frame = tk.Frame(dialog)
        btn_frame.grid(row=2, column=0, padx=12, pady=(6, 12), sticky="e")
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="left")

        dialog.protocol("WM_DELETE_WINDOW", on_cancel)
        dialog.bind("<Return>", lambda _e: on_ok())
        dialog.bind("<Escape>", lambda _e: on_cancel())
        dialog.wait_window()
        root.destroy()
        return result["value"]

    def save_csv(self, _event):
        if not self.saved_samples:
            self._set_status("No samples to save")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(SAVE_DIR, f"glove_realtime_{timestamp}.csv")
        headers = ["sample_index", "elapsed_sec"] + [f"I{i}" for i in range(CHANNEL_COUNT)]

        with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for row in self.saved_samples:
                writer.writerow(row)

        self._set_status(f"CSV saved: {out_path}")

    def save_heatmap(self, _event):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(SAVE_DIR, f"heatmap_{timestamp}.png")
        self.fig.canvas.draw()
        heatmap_extent = self.ax_map.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(out_path, dpi=200, bbox_inches=heatmap_extent)
        self._set_status(f"Heatmap saved: {out_path}")

    def read_available_frames(self):
        if not self.connected or self.serial_port is None:
            return []

        frames = []
        try:
            waiting = self.serial_port.in_waiting
            raw = self.serial_port.read(waiting if waiting > 0 else 1)
        except Exception as exc:
            self.disconnect_serial()
            self._set_status(f"Read failed: {exc}")
            return []

        if not raw:
            return frames

        self.serial_text_buffer += raw.decode("utf-8", errors="ignore")
        lines = self.serial_text_buffer.splitlines()
        if self.serial_text_buffer and not self.serial_text_buffer.endswith(("\n", "\r")):
            self.serial_text_buffer = lines.pop() if lines else self.serial_text_buffer
        else:
            self.serial_text_buffer = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue
            values = parse_serial_line(line, CHANNEL_COUNT)
            if values is not None:
                frames.append(values)

        return frames

    def _history_matrix(self) -> np.ndarray:
        return np.vstack(self.history)

    def _display_matrix(self, history_matrix: np.ndarray) -> np.ndarray:
        if self.baseline is not None:
            return history_matrix - self.baseline[None, :]
        return history_matrix

    def _get_heatmap_values(self) -> np.ndarray:
        if self.baseline is not None and HEATMAP_MODE == "delta":
            delta = self.latest_values - self.baseline
            if DELTA_POSITIVE_ONLY:
                delta = np.maximum(delta, 0.0)
            delta[np.abs(delta) < HEATMAP_DEADBAND] = 0.0
            return self._apply_heatmap_dynamics(delta)
        return self._apply_heatmap_dynamics(self.latest_values)

    def _apply_heatmap_dynamics(self, target_values: np.ndarray) -> np.ndarray:
        target_values = np.asarray(target_values, dtype=float)
        rising = target_values >= self.display_heatmap_values
        alphas = np.where(rising, HEATMAP_ATTACK_ALPHA, HEATMAP_RELEASE_ALPHA)
        self.display_heatmap_values = (
            alphas * target_values + (1.0 - alphas) * self.display_heatmap_values
        )
        release_mask = target_values < HEATMAP_RELEASE_TO_ZERO_THRESHOLD
        self.display_heatmap_values[release_mask] = 0.0
        self.display_heatmap_values[np.abs(self.display_heatmap_values) < HEATMAP_ZERO_HOLD_THRESHOLD] = 0.0
        return self.display_heatmap_values.copy()

    def _get_heatmap_limits(self, heatmap_values: np.ndarray) -> Tuple[float, float]:
        if not USE_AUTO_RANGE:
            return GLOBAL_MIN, GLOBAL_MAX

        if not HEATMAP_USE_HISTORY_RANGE:
            if self.baseline is not None and HEATMAP_MODE == "delta" and DELTA_POSITIVE_ONLY:
                vmin = 0.0
            else:
                vmin = float(np.nanmin(heatmap_values))
            vmax = float(np.nanmax(heatmap_values))
            if vmax - vmin < HEATMAP_MIN_SPAN:
                vmax = vmin + HEATMAP_MIN_SPAN

            self.heatmap_value_min = (
                HEATMAP_RANGE_SMOOTHING * self.heatmap_value_min
                + (1.0 - HEATMAP_RANGE_SMOOTHING) * vmin
            )
            self.heatmap_value_max = (
                HEATMAP_RANGE_SMOOTHING * self.heatmap_value_max
                + (1.0 - HEATMAP_RANGE_SMOOTHING) * vmax
            )
            if self.heatmap_value_max <= self.heatmap_value_min:
                self.heatmap_value_max = self.heatmap_value_min + HEATMAP_MIN_SPAN
            return self.heatmap_value_min, self.heatmap_value_max

        history_matrix = self._history_matrix()
        if self.baseline is not None and HEATMAP_MODE == "delta":
            history_matrix = history_matrix - self.baseline[None, :]
            if DELTA_POSITIVE_ONLY:
                history_matrix = np.maximum(history_matrix, 0.0)
            history_matrix[np.abs(history_matrix) < HEATMAP_DEADBAND] = 0.0

        vmin = float(np.nanpercentile(history_matrix, 5))
        vmax = float(np.nanpercentile(history_matrix, 95))
        if np.isclose(vmin, vmax):
            margin = max(abs(heatmap_values).max(), 1.0)
            if self.baseline is not None and HEATMAP_MODE == "delta":
                vmin, vmax = -margin, margin
            else:
                vmin, vmax = 0.0, margin
        if vmax <= vmin:
            vmax = vmin + 1.0
        return vmin, vmax

    def update_plots(self):
        history_matrix = self._history_matrix()
        display_matrix = self._display_matrix(history_matrix)
        stacked_matrix = display_matrix * WAVE_GAIN + self.channel_offsets[None, :]
        ymin = float(np.nanmin(stacked_matrix))
        ymax = float(np.nanmax(stacked_matrix))
        if np.isclose(ymin, ymax):
            ymax = ymin + 1.0

        for idx, line in enumerate(self.lines):
            line.set_ydata(stacked_matrix[:, idx])

        pad = max(STACKED_SPACING, 0.05 * (ymax - ymin))
        self.ax_sig.set_ylim(ymin - pad, ymax + pad)

        heatmap_values = self._get_heatmap_values()
        value_min, value_max = self._get_heatmap_limits(heatmap_values)
        heatmap_img = render_heatmap_image(
            self.layout_img_bgr,
            self.contours,
            self.contour_channels,
            self.contour_centers,
            heatmap_values,
            self.cmap,
            value_min,
            value_max,
            draw_numbers=DRAW_HEATMAP_NUMBERS_LIVE,
        )
        self.im_artist.set_data(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
        self.fig.canvas.draw_idle()

    def on_timer(self):
        if self.paused:
            return

        now = time.time()
        if now - self.last_frame_time < REFRESH_MS / 1000.0:
            return
        self.last_frame_time = now

        frames = self.read_available_frames()
        if not frames:
            return

        if len(frames) > MAX_RENDER_FRAMES_PER_UPDATE:
            frames = frames[-MAX_RENDER_FRAMES_PER_UPDATE:]

        for values in frames:
            self.latest_values = values
            self.history.append(values.copy())
            self.sample_count += 1
            elapsed = 0.0 if self.start_time is None else (time.time() - self.start_time)
            self.saved_samples.append([self.sample_count, f"{elapsed:.6f}", *values.tolist()])
            if not self.auto_baseline_done:
                self.auto_baseline_sum += values
                self.auto_baseline_count += 1
                if self.auto_baseline_count >= int(AUTO_BASELINE_SECONDS * SAMPLE_RATE_HZ):
                    self.baseline = self.auto_baseline_sum / max(self.auto_baseline_count, 1)
                    self.auto_baseline_done = True

        state = "Connected"
        self._set_status(state)
        self.update_plots()

    def run(self):
        self._set_status("Disconnected")
        try:
            plt.show()
        finally:
            self.disconnect_serial()


def main():
    app = RealTimeGloveApp()
    app.run()


if __name__ == "__main__":
    main()
