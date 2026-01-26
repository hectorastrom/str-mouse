# @Time    : 2026-01-14 12:52
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : utils.py

import torch as t
import torch.nn.functional as F
import pandas as pd
import numpy as np
from threading import Event, Lock
from datetime import datetime
import time
import re
import os


RAW_MOUSE_DATA_DIR = "raw_mouse_data"


def generate_trial_filename() -> str:
    """
    Generate a unique trial filename using timestamp (which is unique, compared
    to our old approach of using a counter)
    
    Format: trial_DD-HH-MM-SS-mmm.csv (day-hour-minute-second-millisecond)
    """
    now = datetime.now()
    return f"trial_{now.day:02d}-{now.hour:02d}-{now.minute:02d}-{now.second:02d}-{now.microsecond // 1000:03d}.csv"

def build_char_map():
    """
    Return a dict of {logit idx : char}
    
    Contains a-z (0-25), A-Z (26-51), and ' ' (52)
    """
    lower_map = {i: chr(ord("a") + i) for i in range(26)}  # lowercase 0-25
    upper_map = {i + 26: chr(ord("A") + i) for i in range(26)}  # uppercase 26-51
    char_map = lower_map | upper_map
    char_map[52] = " "  # space at index 52
    return char_map

def build_inverse_char_map():
    """
    Return dict of {char : logit idx}
    
    Contains 0-52, where 0-25 are a-z, 26-51 are A-Z, and 52 is ' '
    """
    lower_map = {chr(i + ord('a')): i for i in range(26)}  # a-z -> 0-25
    upper_map = {chr(i + ord("A")): i + 26 for i in range(26)}  # A-Z -> 26-51
    inverted_map = lower_map | upper_map
    inverted_map[' '] = 52  # space at index 52
    return inverted_map


def char_to_folder_name(char: str) -> str:
    """
    Convert a character to its folder name for raw_mouse_data storage.
    
    Mac filesystems are case-insensitive, so uppercase uses uA, uB, etc. prefix.
    """
    if char == ' ':
        return 'space'
    elif char.isupper():
        return f"u{char}"
    else:  # lowercase
        return char


def folder_name_to_char(folder: str) -> str:
    """
    Convert a folder name back to its character.
    
    Inverse of char_to_folder_name.
    """
    if folder == 'space':
        return ' '
    elif folder.startswith('u') and len(folder) == 2:
        return folder[1]  # uppercase letter
    else:  # lowercase
        return folder


def delete_sample(char: str, filename: str) -> bool:
    """
    Delete a sample file for a given character
    
    Args:
        char: The character whose sample to delete (e.g., 'a', 'A', ' ')
        filename: The filename of the trial to delete (e.g., 'trial_15-10-30-45-123.csv')
    
    Returns:
        True if file was deleted, False if file didn't exist
    """
    import os
    folder = char_to_folder_name(char)
    filepath = os.path.join(RAW_MOUSE_DATA_DIR, folder, filename)
    
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False


def csv_to_numpy(filepath, samples_last=True) -> np.ndarray:
    """
    Convert a CSV file of mouse stroke data into a (2, T) (or (T, 2) if
    samples_last is False) nparray of positions
    """
    df = pd.read_csv(filepath)
    # accumulate pos
    df["pos_x"] = df["velocity_x"].cumsum().astype("float64")
    df["pos_y"] = df["velocity_y"].cumsum().astype("float64")
    del df["velocity_x"]
    del df["velocity_y"]
    del df["timestamp"]
    full_data = df.to_numpy()
    if samples_last: full_data = full_data.T # (2, T)
    return full_data


def load_chars(root_dir: str, max_trials=-1, samples_last=True, return_tensor=True, silent=False):
    """
    Load all labeled character trials from CSV files under a directory.

    Expected directory layout:
        root_dir/
            *.csv

    Each CSV corresponds to one trial (one variable-length time series).

    Returns:
        (trial_ids, trials)

        trial_ids:
            List[str]
            Filenames for each loaded trias aligned with `trials`

        trials:
            List[torch.Tensor] if return_tensor is True else List[np.ndarray]
            A list of tensors/arrays, one per trial. Each array is a time series with shape:
                - (T, 2) if samples_last is False
                - (2, T) if samples_last is True

            Notes:
            - Trials may have different lengths T, so this function intentionally returns
              a list rather than a single stacked tensor.

    Args:
        root_dir:
            Directory containing trial CSV files.
        max_trials:
            Optional cap on how many trials to load. Leave as -1 to load all trials
        samples_last:
            Controls whether the returned tensors store time in the first dimension
            (False) or the last dimension (True).
        return_tensor:
            Returns torch.Tensor if True, or np.ndarray if False
        silent:
            If True, suppress per-file logging.
    """
    filenames = []
    arrays = []
    if not silent:
        print(f"Found {len(os.listdir(root_dir))} trials in root_dir {root_dir}")
        if max_trials > 0:
            print(f"Limiting to most recent {max_trials} trials.")

    # get all csv files and sort by legacy files (trial_##.csv) first, then timestamp files
    def sort_key(filename):
        match = re.match(r"trial_(\d+)\.csv$", filename)
        if match:
            # legacy file numbered
            return (0, int(match.group(1)), filename)
        else:
            # new timestamp file
            return (1, 0, filename)

    csv_files = sorted(
        [f for f in os.listdir(root_dir) if f.endswith(".csv")], key=sort_key
    )

    if max_trials > 0:
        csv_files = csv_files[-max_trials:]

    for filename in csv_files:
        filepath = os.path.join(root_dir, filename)
        filenames.append(filename)
        array = csv_to_numpy(filepath, samples_last=samples_last)
        if return_tensor:
            array = t.from_numpy(array).to(dtype=t.float32)
        arrays.append(array)

    if not silent:
        print(f"Loaded {len(arrays)} characters from root_dir {root_dir}")
    return (filenames, arrays)


def velocities_to_positions(velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert velocity data to position data via cumulative sum.
    
    Args:
        velocities: np.ndarray of shape (N, 2) with [velocity_x, velocity_y] columns
    
    Returns:
        Tuple of (x_positions, y_positions), each of shape (N,)
    """
    if velocities.size == 0:
        return np.array([]), np.array([])
    
    pos_x = np.cumsum(velocities[:, 0]).astype(np.float64)
    pos_y = np.cumsum(velocities[:, 1]).astype(np.float64)
    return pos_x, pos_y


def build_img(
    x: t.Tensor | np.ndarray,
    y: t.Tensor | np.ndarray,
    downsample_size: int = 28,
    padding: int = 4,
    stroke_width: int = 1,
    invert_colors: bool = False,
):
    """
    Preprocess raw (x, y) stroke data into a downsampled image.

    Preprocessing goals:
      - Preserve aspect ratio and center the stroke in-frame
      - Add linear interpolation between samples to fill gaps
      - Apply white stroke on black background

    Args:
        x: Tensor of shape (T,) representing x-coordinates of stroke
        y: Tensor of shape (T,) representing y-coordinates of stroke
        downsample_size: Size (H and W) of output image
        padding: Number of pixels to pad around the stroke
        stroke_width: Width of the stroke in pixels
        invert_colors: If True, use white stroke on black background

    Returns:
      Tensor of shape (H, W) with values in [0, 1].
    """

    if isinstance(x, np.ndarray) and isinstance(x, np.ndarray):
        x = t.from_numpy(x).to(dtype=t.float32)
        y = t.from_numpy(y).to(dtype=t.float32)

    bg_color = 1.0 if invert_colors else 0.0
    fg_color = 1 - bg_color

    if x.numel() < 2 or y.numel() < 2:
        # not a real stroke: return blank canvas
        return t.full(
            (downsample_size, downsample_size),
            float(bg_color),
            dtype=t.float32,
            device=x.device,
        )

    x = x.to(dtype=t.float32)
    y = y.to(dtype=t.float32)

    # find bounding box in original coordinate space
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    span_x = (x_max - x_min).clamp_min(1e-6)
    span_y = (y_max - y_min).clamp_min(1e-6)

    # Uniform scaling: preserve aspect ratio
    # We map the larger total span to the smaller designated drawable region
    size = int(downsample_size)
    pad = int(padding)
    drawable = max(1, size - 1 - 2 * pad)
    scale = drawable / t.maximum(span_x, span_y)

    # Center stroke by translating to its midpoint, then into image center
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    x_s = (x - cx) * scale
    y_s = (y - cy) * scale

    # Image coordinates: origin top-left, y increases downward. y used as row
    center = (size - 1) * 0.5
    x_px_f = x_s + center
    y_px_f = y_s + center

    # Clamp to image bounds (floats)
    x_px_f = x_px_f.clamp(0.0, float(size - 1))
    y_px_f = y_px_f.clamp(0.0, float(size - 1))

    # build canvas
    img = t.full((size, size), float(bg_color), dtype=t.float32, device=x.device)

    # linear interpolation btwn points
    assert x.numel() == y.numel(), f"Length mismatch: x={x.numel()} y={y.numel()}"
    for i in range(x_px_f.numel() - 1):
        x0, y0 = x_px_f[i], y_px_f[i]
        x1, y1 = x_px_f[i + 1], y_px_f[i + 1]

        dx = x1 - x0
        dy = y1 - y0

        steps = int(t.maximum(dx.abs(), dy.abs()).item()) + 1
        if steps <= 1:
            xi = int(round(float(x0.item())))
            yi = int(round(float(y0.item())))
            img[yi, xi] = fg_color
            continue

        ts = t.linspace(0.0, 1.0, steps=steps, device=x.device, dtype=t.float32)
        xs = (x0 + ts * dx).round().to(dtype=t.int64)
        ys = (y0 + ts * dy).round().to(dtype=t.int64)

        xs = xs.clamp(0, size - 1)
        ys = ys.clamp(0, size - 1)

        img[ys, xs] = fg_color

    # Thicken stroke with maxpool2d
    if stroke_width > 0:
        ink = (
            (img != bg_color).to(dtype=t.float32).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, H, W)
        k = 2 * int(stroke_width) + 1
        ink = (
            F.max_pool2d(ink, kernel_size=k, stride=1, padding=int(stroke_width))
            .squeeze(0)
            .squeeze(0)
        )

        # Apply stroke to canvas
        img = t.full((size, size), float(bg_color), dtype=t.float32, device=x.device)
        img[ink > 0.0] = fg_color

    return img  # (H, W)


class GlobalInputManager:
    """
    Maintains a single persistent connection to macOS input events to avoid
    initializing multiple event listeners which cause lag / crashes.
    
    Handles mouse velocity recording with configurable sampling rate.
    """

    def __init__(self, sampling_rate: int = 100):
        """
        Initialize the global input manager.
        
        Args:
            sampling_rate: Sampling rate of mouse in Hz (default 100Hz)
        """
        from pynput import keyboard, mouse # needed to protect headless services
        
        self.sampling_rate = sampling_rate
        self.sample_interval = 1.0 / sampling_rate
        
        # recording state
        self.recording_active = False
        self.recording_running = False
        
        # position tracking for velocity calculation
        self.current_pos = (0, 0)
        self.last_logged_pos = (0, 0)
        
        # recorded data storage
        self.recorded_velocities: list[tuple[str, float, float]] = []  # (timestamp, vx, vy)
        
        # locks and events
        self.first_move_event = Event()
        self.space_pressed_event = Event()
        self.state_lock = Lock()
        self.velocity_lock = Lock()
        self.last_move_time = 0.0

        # start global listeners immediately
        self.mouse_listener = mouse.Listener(on_move=self._on_global_move)
        self.kb_listener = keyboard.Listener(on_press=self._on_global_press)

        self.mouse_listener.start()
        self.kb_listener.start()

        # wait for mac permissions
        time.sleep(1.0)

    def _on_global_move(self, x, y):
        """Handle mouse move events globally."""
        current_time = time.time()
        
        # constantly update last move time, even when not recording
        with self.state_lock:
            self.last_move_time = current_time

        # pass data if we're actively recording
        if self.recording_active:
            if not self.first_move_event.is_set():
                # avoid initial velocity spike by setting initial pos to cursor's initial location
                with self.velocity_lock:
                    self.current_pos = (x, y)
                    self.last_logged_pos = (x, y)
                self.first_move_event.set()
            else:
                with self.velocity_lock:
                    self.current_pos = (x, y)

    def _on_global_press(self, key):
        """Handle key press events globally."""
        if key == keyboard.Key.space:
            self.space_pressed_event.set()

    def _calculate_velocity(self) -> tuple[float, float]:
        """Calculate velocity since last logged position."""
        with self.velocity_lock:
            dx = self.current_pos[0] - self.last_logged_pos[0]
            dy = self.current_pos[1] - self.last_logged_pos[1]
            self.last_logged_pos = self.current_pos
            return dx, dy

    def _log_velocity_loop(self):
        """Background thread that logs velocity at the configured sampling rate."""
        from datetime import datetime
        
        while self.recording_running:
            timestamp = datetime.now().isoformat()
            velocity_x, velocity_y = self._calculate_velocity()
            
            with self.state_lock:
                self.recorded_velocities.append((timestamp, velocity_x, velocity_y))
            
            time.sleep(self.sample_interval)

    def wait_for_space(self):
        """Block until space key is pressed."""
        self.space_pressed_event.clear()
        self.space_pressed_event.wait()

    def start_recording(self):
        """
        Start recording mouse velocities.
        
        Recording begins when the mouse first moves after this call.
        Use stop_recording() to end and retrieve the data.
        """
        from threading import Thread
        
        # reset state
        with self.state_lock:
            self.recorded_velocities = []
            self.last_move_time = time.time()
        
        with self.velocity_lock:
            self.current_pos = (0, 0)
            self.last_logged_pos = (0, 0)
        
        self.first_move_event.clear()
        self.recording_active = True
        self.recording_running = True
        
        # start logging thread
        self._log_thread = Thread(target=self._log_velocity_loop, daemon=True)
        self._log_thread.start()

    def wait_for_first_move(self):
        """Block until the first mouse movement is detected after recording started."""
        while not self.first_move_event.is_set():
            time.sleep(0.01)

    def stop_recording(self) -> np.ndarray:
        """
        Stop recording and return the recorded data as a numpy array.
        
        Returns:
            np.ndarray of shape (N, 2) where N is number of samples,
            columns are [velocity_x, velocity_y]
        """
        self.recording_running = False
        self.recording_active = False
        
        if hasattr(self, '_log_thread'):
            self._log_thread.join(timeout=1.0)
        
        with self.state_lock:
            if not self.recorded_velocities:
                return np.array([]).reshape(0, 2)
            
            # extract just velocities (skip timestamps for the array)
            velocities = [(vx, vy) for _, vx, vy in self.recorded_velocities]
            return np.array(velocities, dtype=np.float64)

    def get_recorded_data_with_timestamps(self) -> list[tuple[str, float, float]]:
        """
        Get the full recorded data including timestamps.
        
        Returns:
            List of (timestamp, velocity_x, velocity_y) tuples
        """
        with self.state_lock:
            return list(self.recorded_velocities)

    def save_to_csv(self, filename: str):
        """
        Save the recorded velocity data to a CSV file.
        
        Args:
            filename: Path to save the CSV file
        """
        import csv
        
        with self.state_lock:
            data = list(self.recorded_velocities)
        
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "velocity_x", "velocity_y"])
            for timestamp, vx, vy in data:
                writer.writerow([timestamp, vx, vy])

    def get_time_since_last_move(self) -> float:
        """Get seconds elapsed since last mouse movement."""
        with self.state_lock:
            return time.time() - self.last_move_time

    def shutdown(self):
        """Stop all listeners and clean up."""
        self.recording_running = False
        self.recording_active = False
        self.mouse_listener.stop()
        self.kb_listener.stop()


if __name__ == "__main__":
    print(build_inverse_char_map())
