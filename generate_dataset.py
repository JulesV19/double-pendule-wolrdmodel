"""
Double pendulum dataset generator for world model training.

Output: dataset/double_pendulum/traj_XXXX.npz
Each file contains:
  - frames: (T, H, W, 3) uint8  — rendered frames
  - states: (T, 4) float64      — [theta1, theta2, omega1, omega2]
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import time


# ── Physics ────────────────────────────────────────────────────────────────────

def _derivatives(state, L1, L2, m1, m2, g):
    theta1, theta2, omega1, omega2 = state
    d = theta2 - theta1
    cos_d, sin_d = np.cos(d), np.sin(d)

    denom1 = (m1 + m2) * L1 - m2 * L1 * cos_d**2
    denom2 = (L2 / L1) * denom1

    domega1 = (
        m2 * L1 * omega1**2 * sin_d * cos_d
        + m2 * g * np.sin(theta2) * cos_d
        + m2 * L2 * omega2**2 * sin_d
        - (m1 + m2) * g * np.sin(theta1)
    ) / denom1

    domega2 = (
        -m2 * L2 * omega2**2 * sin_d * cos_d
        + (m1 + m2) * g * np.sin(theta1) * cos_d
        - (m1 + m2) * L1 * omega1**2 * sin_d
        - (m1 + m2) * g * np.sin(theta2)
    ) / denom2

    return np.array([omega1, omega2, domega1, domega2])


def simulate(n_frames, dt, rng, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
    state = np.array([
        rng.uniform(-np.pi, np.pi),   # theta1
        rng.uniform(-np.pi, np.pi),   # theta2
        rng.uniform(-2.0, 2.0),       # omega1
        rng.uniform(-2.0, 2.0),       # omega2
    ])

    states = np.empty((n_frames, 4))
    states[0] = state

    for i in range(1, n_frames):
        k1 = _derivatives(state, L1, L2, m1, m2, g)
        k2 = _derivatives(state + dt / 2 * k1, L1, L2, m1, m2, g)
        k3 = _derivatives(state + dt / 2 * k2, L1, L2, m1, m2, g)
        k4 = _derivatives(state + dt * k3, L1, L2, m1, m2, g)
        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        states[i] = state

    return states


# ── Rendering ──────────────────────────────────────────────────────────────────

def render_frame(state, img_size=64, L1=1.0, L2=1.0):
    img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    theta1, theta2 = state[0], state[1]

    cx = cy = img_size / 2
    scale = img_size * 0.22          # fits total arm length (2 units) with margin

    x1 = cx + scale * L1 * np.sin(theta1)
    y1 = cy + scale * L1 * np.cos(theta1)
    x2 = x1 + scale * L2 * np.sin(theta2)
    y2 = y1 + scale * L2 * np.cos(theta2)

    draw.line([(cx, cy), (x1, y1)], fill=(255, 255, 255), width=2)
    draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255), width=2)

    r = max(2, img_size // 22)
    draw.ellipse([(cx - 2, cy - 2), (cx + 2, cy + 2)], fill=(160, 160, 160))
    draw.ellipse([(x1 - r, y1 - r), (x1 + r, y1 + r)], fill=(255, 255, 255))
    draw.ellipse([(x2 - r, y2 - r), (x2 + r, y2 + r)], fill=(255, 255, 255))

    return np.array(img, dtype=np.uint8)


def render_trajectory(states, img_size=64, L1=1.0, L2=1.0):
    return np.stack([render_frame(s, img_size, L1, L2) for s in states])


# ── Dataset generation ─────────────────────────────────────────────────────────

def generate_dataset(
    n_trajectories: int = 1000,
    n_frames: int = 50,
    img_size: int = 64,
    dt: float = 0.05,
    output_dir: str = "dataset/double_pendulum",
    seed: int = 42,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    t0 = time.time()

    for i in range(n_trajectories):
        states = simulate(n_frames, dt, rng)
        frames = render_trajectory(states, img_size)

        np.savez_compressed(
            out / f"traj_{i:04d}.npz",
            frames=frames,   # (T, H, W, 3) uint8
            states=states,   # (T, 4) float64: theta1, theta2, omega1, omega2
        )

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_trajectories - i - 1) / rate
            print(f"  {i+1:4d}/{n_trajectories}  |  {elapsed:.0f}s elapsed  |  ~{remaining:.0f}s remaining")

    elapsed = time.time() - t0
    total_frames = n_trajectories * n_frames
    size_mb = sum(f.stat().st_size for f in out.glob("*.npz")) / 1e6

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Trajectories : {n_trajectories}")
    print(f"  Frames/traj  : {n_frames}")
    print(f"  Resolution   : {img_size}x{img_size}")
    print(f"  Total frames : {total_frames:,}")
    print(f"  Dataset size : {size_mb:.1f} MB")
    print(f"  Output       : {out.resolve()}")


if __name__ == "__main__":
    generate_dataset(
        n_trajectories=1000,
        n_frames=50,
        img_size=64,
        dt=0.05,
        output_dir="dataset/double_pendulum",
        seed=42,
    )
