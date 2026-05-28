"""
Simple pendulum dataset generator for world model training.

Output: dataset/double_pendulum/traj_XXXX.npz
Each file contains:
  - frames: (T, H, W, 3) uint8  — rendered frames
  - states: (T, 2) float64      — [theta, omega]
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import time


# ── Physics ────────────────────────────────────────────────────────────────────

def _derivatives(state, L, g=9.81):
    theta, omega = state
    return np.array([omega, -(g / L) * np.sin(theta)])


def simulate(n_frames, dt, rng, L=1.0, g=9.81):
    # Conditions initiales : oscillation garantie (énergie < énergie au sommet)
    # E_max = m*g*L*2  (sommet), E = 0.5*omega^2 - g/L*cos(theta)
    # Pour osciller : 0.5*omega^2 < g/L*(1 + cos(theta))
    while True:
        theta0 = rng.uniform(-np.pi * 0.9, np.pi * 0.9)
        omega0 = rng.uniform(-4.0, 4.0)
        energy = 0.5 * omega0 ** 2 - g / L * np.cos(theta0)
        if energy < g / L:   # pendule oscillant (pas de rotation complète)
            break

    state  = np.array([theta0, omega0])
    states = np.empty((n_frames, 2))
    states[0] = state

    for i in range(1, n_frames):
        k1 = _derivatives(state, L, g)
        k2 = _derivatives(state + dt / 2 * k1, L, g)
        k3 = _derivatives(state + dt / 2 * k2, L, g)
        k4 = _derivatives(state + dt * k3, L, g)
        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        states[i] = state

    return states


# ── Rendering ──────────────────────────────────────────────────────────────────

def render_frame(state, img_size=64, L=1.0):
    img  = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    theta = state[0]
    cx = cy = img_size / 2
    scale = img_size * 0.40          # bras plus grand : 1 seul bras

    x1 = cx + scale * L * np.sin(theta)
    y1 = cy + scale * L * np.cos(theta)

    draw.line([(cx, cy), (x1, y1)], fill=(255, 255, 255), width=2)

    r = max(2, img_size // 22)
    draw.ellipse([(cx - 2, cy - 2), (cx + 2, cy + 2)], fill=(160, 160, 160))
    draw.ellipse([(x1 - r, y1 - r), (x1 + r, y1 + r)], fill=(255, 255, 255))

    return np.array(img, dtype=np.uint8)


def render_trajectory(states, img_size=64, L=1.0):
    return np.stack([render_frame(s, img_size, L) for s in states])


# ── Dataset generation ─────────────────────────────────────────────────────────

def generate_dataset(
    n_trajectories: int = 1000,
    n_frames:       int = 50,
    img_size:       int = 64,
    dt:             float = 0.05,
    output_dir:     str = "dataset/double_pendulum",
    seed:           int = 42,
):
    out = Path(output_dir)
    if out.exists():
        for f in out.glob("traj_*.npz"):
            f.unlink()
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    t0  = time.time()

    for i in range(n_trajectories):
        states = simulate(n_frames, dt, rng)
        frames = render_trajectory(states, img_size)

        np.savez_compressed(
            out / f"traj_{i:04d}.npz",
            frames=frames,   # (T, H, W, 3) uint8
            states=states,   # (T, 2) float64: theta, omega
        )

        if (i + 1) % 100 == 0:
            elapsed   = time.time() - t0
            rate      = (i + 1) / elapsed
            remaining = (n_trajectories - i - 1) / rate
            print(f"  {i+1:4d}/{n_trajectories}  |  {elapsed:.0f}s elapsed  |  ~{remaining:.0f}s remaining")

    elapsed      = time.time() - t0
    total_frames = n_trajectories * n_frames
    size_mb      = sum(f.stat().st_size for f in out.glob("*.npz")) / 1e6

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Trajectories : {n_trajectories}")
    print(f"  Frames/traj  : {n_frames}")
    print(f"  Resolution   : {img_size}x{img_size}")
    print(f"  Total frames : {total_frames:,}")
    print(f"  Dataset size : {size_mb:.1f} MB")
    print(f"  Output       : {out.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate simple pendulum dataset.")
    parser.add_argument("--n_trajectories", type=int,   default=2000)
    parser.add_argument("--n_frames",       type=int,   default=500)
    parser.add_argument("--img_size",       type=int,   default=64)
    parser.add_argument("--dt",             type=float, default=0.05)
    parser.add_argument("--output_dir",     type=str,   default="dataset/double_pendulum")
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    generate_dataset(
        n_trajectories=args.n_trajectories,
        n_frames=args.n_frames,
        img_size=args.img_size,
        dt=args.dt,
        output_dir=args.output_dir,
        seed=args.seed,
    )
