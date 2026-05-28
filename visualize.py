"""
Interactive viewer for the simple pendulum dataset.

Usage:
  python3 visualize.py                  # random trajectory
  python3 visualize.py --idx 42         # specific trajectory
  python3 visualize.py --n 4            # grid of 4 random trajectories
  python3 visualize.py --save           # export as GIF
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path


DATASET_DIR = Path("dataset/pendulum")


def load_traj(idx: int):
    path = DATASET_DIR / f"traj_{idx:04d}.npz"
    data = np.load(path)
    return data["frames"], data["states"]


def random_idx(rng=None):
    files = sorted(DATASET_DIR.glob("traj_*.npz"))
    rng = rng or np.random.default_rng()
    return int(rng.integers(len(files)))


# ── Single trajectory ──────────────────────────────────────────────────────────

def view_single(idx: int, save: bool = False):
    frames, states = load_traj(idx)
    n_frames = len(frames)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5),
                             gridspec_kw={"width_ratios": [1, 1.4]})
    fig.patch.set_facecolor("#111")
    for ax in axes:
        ax.set_facecolor("#111")

    # Left: rendered frame
    ax_img = axes[0]
    ax_img.axis("off")
    im = ax_img.imshow(frames[0], interpolation="nearest")
    title = ax_img.set_title(f"traj {idx:04d} — frame 0/{n_frames-1}",
                             color="white", fontsize=9)

    # Right: state plot (angles over time)
    ax_state = axes[1]
    t = np.arange(n_frames) * 0.05
    ax_state.plot(t, states[:, 0], color="#4fc3f7", lw=1, label="θ")
    ax_state.plot(t, states[:, 1], color="#ff8a65", lw=1, label="ω")
    ax_state.set_xlabel("time (s)", color="white", fontsize=8)
    ax_state.set_ylabel("θ (rad) / ω (rad/s)", color="white", fontsize=8)
    ax_state.tick_params(colors="white", labelsize=7)
    for spine in ax_state.spines.values():
        spine.set_edgecolor("#444")
    ax_state.legend(fontsize=8, labelcolor="white",
                    facecolor="#222", edgecolor="#444")
    vline = ax_state.axvline(t[0], color="#fff", lw=0.8, ls="--", alpha=0.6)

    plt.tight_layout(pad=1.2)

    def update(frame):
        im.set_data(frames[frame])
        title.set_text(f"traj {idx:04d} — frame {frame}/{n_frames-1}")
        vline.set_xdata([t[frame], t[frame]])
        return im, title, vline

    interval = 80  # ms between frames
    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=interval, blit=True, repeat=True)

    if save:
        out = f"traj_{idx:04d}.gif"
        ani.save(out, writer="pillow", fps=1000 // interval)
        print(f"Saved {out}")
    else:
        plt.show()


# ── Grid of N trajectories ─────────────────────────────────────────────────────

def view_grid(n: int, save: bool = False):
    rng = np.random.default_rng()
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    indices = [random_idx(rng) for _ in range(n)]
    all_frames = [load_traj(i)[0] for i in indices]
    n_frames = min(len(f) for f in all_frames)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    fig.patch.set_facecolor("#111")
    axes = np.array(axes).flatten()

    ims, titles = [], []
    for k, ax in enumerate(axes):
        ax.set_facecolor("#111")
        ax.axis("off")
        if k < n:
            im = ax.imshow(all_frames[k][0], interpolation="nearest")
            t = ax.set_title(f"#{indices[k]:04d}", color="white", fontsize=8)
            ims.append(im)
            titles.append(t)

    plt.tight_layout(pad=0.5)

    def update(frame):
        for k, (im, title) in enumerate(zip(ims, titles)):
            im.set_data(all_frames[k][frame])
        return ims

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=80, blit=True, repeat=True)

    if save:
        out = "grid.gif"
        ani.save(out, writer="pillow", fps=12)
        print(f"Saved {out}")
    else:
        plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=None, help="trajectory index")
    parser.add_argument("--n",   type=int, default=1,    help="grid size (>1 → grid mode)")
    parser.add_argument("--save", action="store_true",   help="export as GIF")
    args = parser.parse_args()

    if args.n > 1:
        view_grid(args.n, save=args.save)
    else:
        idx = args.idx if args.idx is not None else random_idx()
        view_single(idx, save=args.save)
