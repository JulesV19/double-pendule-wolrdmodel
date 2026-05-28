"""
Navigateur interactif du dataset pendule simple.

Affiche pour chaque trajectoire :
  • Frame animée + canal diff (signal de mouvement pour frame stacking)
  • Courbes θ, ω avec curseur temporel
  • Portrait de phase θ vs ω

Contrôles :
  ◀ Prev / Next ▶  — changer de trajectoire
  ⏸ / ▶           — pause / lecture
  Slider Frame     — scrubbing manuel

Usage:
  python3 browse_dataset.py
  python3 browse_dataset.py --dataset-dir dataset/pendulum --fps 15
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
from pathlib import Path


DARK   = "#111"
DARK2  = "#1a1a1a"
EDGE   = "#333"
COLORS = ["#4fc3f7", "#ff8a65", "#a5d6a7", "#ce93d8"]   # θ1 θ2 ω1 ω2


# ── Données ────────────────────────────────────────────────────────────────────

def load_traj(path: Path):
    d = np.load(path)
    frames = d["frames"]                              # (T, H, W, 3) uint8
    states = d["states"].astype(np.float32)          # (T, 2) float32: [theta, omega]
    diffs  = np.zeros_like(frames, dtype=np.float32)
    diffs[1:] = frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)
    diffs = np.clip(diffs / 255.0, -1.0, 1.0)
    return frames, states, diffs


# ── Browser ────────────────────────────────────────────────────────────────────

class DatasetBrowser:
    def __init__(self, dataset_dir: str, fps: int = 20):
        self.files = sorted(Path(dataset_dir).glob("traj_*.npz"))
        assert self.files, f"Aucune trajectoire dans {dataset_dir}"
        self.fps     = fps
        self.idx     = 0
        self.t       = 0
        self.playing = True

        self._load()
        self._build_figure()
        self._start()

    # ── Chargement ─────────────────────────────────────────────────────────────

    def _load(self):
        self.frames, self.states, self.diffs = load_traj(self.files[self.idx])
        self.T = len(self.frames)
        self.t = 0

    # ── Figure ─────────────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 8), facecolor=DARK)
        self.fig.patch.set_facecolor(DARK)

        outer = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[10, 1],
            hspace=0.08,
            left=0.04, right=0.98, top=0.94, bottom=0.06,
        )

        # Rangée principale : 5 colonnes
        gs = gridspec.GridSpecFromSubplotSpec(
            2, 5, subplot_spec=outer[0],
            hspace=0.45, wspace=0.38,
            width_ratios=[1.2, 1.2, 1, 1, 1.3],
        )

        self.ax_frame  = self.fig.add_subplot(gs[0, 0])
        self.ax_diff   = self.fig.add_subplot(gs[1, 0])
        self.ax_th     = self.fig.add_subplot(gs[0, 1])
        self.ax_om     = self.fig.add_subplot(gs[1, 1])
        self.ax_phase  = self.fig.add_subplot(gs[:, 3])
        self.ax_info   = self.fig.add_subplot(gs[:, 4])

        # Ligne de contrôles
        ctrl = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=outer[1],
            wspace=0.3,
            width_ratios=[1, 1, 1, 0.2, 4],
        )
        ax_prev  = self.fig.add_subplot(ctrl[0, 0])
        ax_play  = self.fig.add_subplot(ctrl[0, 1])
        ax_next  = self.fig.add_subplot(ctrl[0, 2])
        ax_slide = self.fig.add_subplot(ctrl[0, 4])

        self.btn_prev = Button(ax_prev, "<  Prev", color="#222", hovercolor="#444")
        self.btn_play = Button(ax_play, "Pause", color="#222", hovercolor="#444")
        self.btn_next = Button(ax_next, "Next  >", color="#222", hovercolor="#444")
        self.slider   = Slider(ax_slide, "Frame", 0, self.T - 1,
                               valinit=0, valstep=1, color="#4fc3f7")

        for btn in (self.btn_prev, self.btn_play, self.btn_next):
            btn.label.set_color("white")
            btn.label.set_fontsize(9)
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")

        self.btn_prev.on_clicked(self._prev)
        self.btn_play.on_clicked(self._toggle_play)
        self.btn_next.on_clicked(self._next)
        self.slider.on_changed(self._on_slider)

        self._style_axes()
        self._init_artists()

    def _style(self, ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(DARK2)
        ax.tick_params(colors="#999", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(EDGE)
        if title:  ax.set_title(title, color="#ccc", fontsize=8, pad=3)
        if xlabel: ax.set_xlabel(xlabel, color="#999", fontsize=7)
        if ylabel: ax.set_ylabel(ylabel, color="#999", fontsize=7)

    def _style_axes(self):
        self._style(self.ax_frame, "Frame")
        self._style(self.ax_diff,  "Diff  (mouvement visible)")
        self._style(self.ax_th,    "θ", "frame", "rad")
        self._style(self.ax_om,    "ω", "frame", "rad/s")
        self._style(self.ax_phase, "Phase  θ vs ω", "θ (rad)", "ω (rad/s)")
        self.ax_info.set_facecolor(DARK2)
        self.ax_info.axis("off")
        for sp in self.ax_info.spines.values():
            sp.set_edgecolor(EDGE)

    # ── Artistes initiaux ───────────────────────────────────────────────────────

    def _init_artists(self):
        T  = self.T
        ts = np.arange(T)
        s  = self.states

        # Frame & diff
        self.im_frame = self.ax_frame.imshow(self.frames[0], interpolation="nearest")
        self.ax_frame.axis("off")
        diff0 = self.diffs[0].mean(axis=-1)
        self.im_diff  = self.ax_diff.imshow(diff0, vmin=-1, vmax=1,
                                             cmap="RdBu_r", interpolation="nearest")
        self.ax_diff.axis("off")

        # Courbes d'état
        self.ln_th, = self.ax_th.plot(ts, s[:, 0], color=COLORS[0], lw=1)
        self.ln_om, = self.ax_om.plot(ts, s[:, 1], color=COLORS[2], lw=1)

        # Curseurs verticaux
        self.vlines = [
            ax.axvline(0, color="white", lw=0.8, alpha=0.6)
            for ax in (self.ax_th, self.ax_om)
        ]

        # Portrait de phase
        self.ln_phase, = self.ax_phase.plot(
            s[:, 0], s[:, 1], color=COLORS[0], lw=0.8, alpha=0.5)
        self.pt_phase, = self.ax_phase.plot([], [], "o", color="white", ms=5, zorder=5)

        # Infobox
        self._draw_info()
        self._update_title()

    # ── Mise à jour frame ───────────────────────────────────────────────────────

    def _update_frame(self, t):
        self.im_frame.set_data(self.frames[t])
        self.im_diff.set_data(self.diffs[t].mean(axis=-1))
        for vl in self.vlines:
            vl.set_xdata([t, t])
        self.pt_phase.set_data([self.states[t, 0]], [self.states[t, 1]])

        # Slider sans récursion
        self.slider.eventson = False
        self.slider.set_val(t)
        self.slider.eventson = True

    # ── Rechargement complet après changement de trajectoire ───────────────────

    def _reload(self):
        T  = self.T
        ts = np.arange(T)
        s  = self.states

        self.ln_th.set_data(ts, s[:, 0])
        self.ln_om.set_data(ts, s[:, 1])

        for ax, col in zip((self.ax_th, self.ax_om), (0, 1)):
            ax.set_xlim(0, T - 1)
            lo, hi = s[:, col].min(), s[:, col].max()
            margin  = max(abs(hi - lo) * 0.12, 0.1)
            ax.set_ylim(lo - margin, hi + margin)

        self.ax_phase.clear()
        self._style(self.ax_phase, "Phase  θ vs ω", "θ (rad)", "ω (rad/s)")
        self.ln_phase, = self.ax_phase.plot(
            s[:, 0], s[:, 1], color=COLORS[0], lw=0.8, alpha=0.5)
        self.pt_phase, = self.ax_phase.plot([], [], "o", color="white", ms=5, zorder=5)

        self.slider.valmax = T - 1
        self.slider.ax.set_xlim(0, T - 1)

        self._draw_info()
        self._update_title()

    # ── Infobox ─────────────────────────────────────────────────────────────────

    def _draw_info(self):
        self.ax_info.clear()
        self.ax_info.axis("off")
        self.ax_info.set_facecolor(DARK2)
        s = self.states

        lines = [
            ("STATS TRAJECTOIRE", "", "white"),
            ("", "", "white"),
            ("θ", f"[{s[:,0].min():.2f}, {s[:,0].max():.2f}] rad", COLORS[0]),
            ("ω", f"[{s[:,1].min():.2f}, {s[:,1].max():.2f}] rad/s", COLORS[2]),
            ("", "", "white"),
            ("Frames", str(self.T), "#aaa"),
            ("dt", "0.05 s", "#aaa"),
            ("Durée", f"{self.T * 0.05:.2f} s", "#aaa"),
            ("", "", "white"),
            ("Diff max", f"{abs(self.diffs).max():.3f}", "#aaa"),
            ("Diff mean", f"{abs(self.diffs).mean():.4f}", "#aaa"),
        ]

        for i, (label, val, color) in enumerate(lines):
            y = 0.97 - i * 0.075
            weight = "bold" if label == "STATS TRAJECTOIRE" else "normal"
            self.ax_info.text(0.05, y, label, transform=self.ax_info.transAxes,
                              color=color, fontsize=8, fontweight=weight,
                              va="top")
            self.ax_info.text(0.55, y, val, transform=self.ax_info.transAxes,
                              color="white", fontsize=8, va="top")

    def _update_title(self):
        n = len(self.files)
        self.fig.suptitle(
            f"Dataset browser  —  trajectoire {self.idx + 1} / {n}"
            f"  ({self.files[self.idx].name})",
            color="white", fontsize=11, y=0.99,
        )

    # ── Animation ───────────────────────────────────────────────────────────────

    def _animate(self, _):
        if self.playing:
            self.t = (self.t + 1) % self.T
            self._update_frame(self.t)
        return []

    def _start(self):
        interval = max(16, int(1000 / self.fps))
        self.anim = animation.FuncAnimation(
            self.fig, self._animate,
            interval=interval, blit=False, cache_frame_data=False,
        )
        plt.show()

    # ── Callbacks ───────────────────────────────────────────────────────────────

    def _prev(self, _):
        self.idx = (self.idx - 1) % len(self.files)
        self._load()
        self._reload()
        self._update_frame(0)

    def _next(self, _):
        self.idx = (self.idx + 1) % len(self.files)
        self._load()
        self._reload()
        self._update_frame(0)

    def _toggle_play(self, _):
        self.playing = not self.playing
        label = "Pause" if self.playing else "Play"
        self.btn_play.label.set_text(label)
        self.fig.canvas.draw_idle()

    def _on_slider(self, val):
        self.t = int(val)
        self._update_frame(self.t)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset/pendulum")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    DatasetBrowser(args.dataset_dir, fps=args.fps)


if __name__ == "__main__":
    main()
