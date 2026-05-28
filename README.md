# LeWorldModel

A clean implementation of **LeWorldModel** (Maes et al., 2026) trained on a simple pendulum. LeWorldModel is a JEPA architecture (LeCun et al.) — the model predicts in latent space rather than pixel space, and learns physical dynamics without any explicit supervision on the state variables (θ, ω).

> **Based on:**
> - Maes et al., *"Le World Model"*, arXiv:2603.19312 (2026) — architecture implemented here
> - LeCun, *"A Path Towards Autonomous Machine Intelligence"* (2022) — JEPA framework
> - Assran et al., *"Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"* (I-JEPA, CVPR 2023)
> - Bardes et al., *"V-JEPA: Latent Video Prediction for Visual Representation"* (V-JEPA, 2024)

## Architecture

```
(frame_t, frame_t − frame_{t−1})   ← 6-channel input (frame stacking)
              ↓
       CNN encoder online   →   z_ctx ∈ R^128   [gradient active]
       CNN encoder target   →   z_tgt ∈ R^128   [EMA, no gradient]
              ↓
       MLP predictor   →   ẑ_{t+k}   (k = 1 … rollout_k)
```

**Loss:**
```
loss = pred_loss + λ · sigreg
pred_loss = (1/K) Σ_{k=1..K} [ cosine_dist(ẑ_{t+k}, z*_{t+k}) + α · MSE(ẑ_{t+k}, z*_{t+k}) ]
```

- `cosine_dist` constrains the direction of z (encodes θ)
- `MSE` constrains the magnitude of z (encodes energy / amplitude)
- `sigreg` ([SIGReg](https://arxiv.org/abs/2603.19312)) forces z ~ N(0, I) via the Epps-Pulley test, preventing representational collapse

The MLP predictor sees only z_t (no temporal context). This forces the encoder to embed ω directly into z — without ω, predicting z_{t+1} from z_t alone is impossible.

**Key hyperparameters:**

| Parameter | Value | Note |
|---|---|---|
| `embed_dim` | 128 | latent dimension |
| `rollout_k` | 20 | ~1s ≈ half-period of the pendulum |
| `lam` | 1.0 | SIGReg weight |
| `mse_coef` | 0.1 | MSE weight in pred_loss |
| `ema_momentum` | 0.996 | target encoder EMA |

## Dataset

Simple pendulum, rendered at 64×64 px.

```bash
python generate_dataset.py --n_trajectories 2000 --n_frames 500
```

Output: `dataset/pendulum/traj_XXXX.npz` — each file contains `frames (T, H, W, 3)` and `states (T, 2)` = [θ, ω].

## Training

**World model** (local):
```bash
python train_lewm.py
python train_lewm.py --lam 1.0 --rollout-k 20 --epochs 50
python train_lewm.py --checkpoint checkpoints/lewm_best.pt   # resume
```

**World model** (Colab): `notebooks/train_colab.ipynb`

**Decoder** z → frame (frozen encoder):
```bash
python train_decoder.py --checkpoint checkpoints/lewm_best.pt
```

**Decoder** (Colab): `notebooks/train_decoder_colab.ipynb`

## Evaluation

```bash
python eval_lewm.py --checkpoint checkpoints/lewm_best.pt
python scatter_probe.py --checkpoint checkpoints/lewm_best.pt
```

`eval_lewm.py` reports: linear/MLP probe R²(θ, ω), uniformity, alignment, prediction horizon cosine similarity.

`scatter_probe.py` outputs a 2×2 scatter grid (θ/ω × linear/MLP probe).

## Visualization

```bash
python imagine.py                   # real vs imagined frames side by side
python browse_dataset.py            # interactive dataset browser (θ, ω curves + phase portrait)
python visualize.py --n 4           # grid of 4 random trajectories
```

## File structure

```
models/
  lewm.py          — LeWorldModel: encoder + EMA + MLP predictor
  encoder.py       — ContextEncoder (6ch) + TargetEncoder (EMA)
  decoder.py       — Decoder z → frame
  sigreg.py        — SIGReg regularizer
notebooks/
  train_colab.ipynb
  train_decoder_colab.ipynb
dataset.py         — PendulumSeqDataset + dataloaders
generate_dataset.py
train_lewm.py
train_decoder.py
eval_lewm.py
scatter_probe.py
imagine.py
browse_dataset.py
visualize.py
```
