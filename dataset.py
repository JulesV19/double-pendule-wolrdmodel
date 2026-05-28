import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


class PendulumFrameDataset(Dataset):
    """
    Returns individual frames (not sequences) for decoder training.
    Each item: tensor (3, H, W) float32 in [0, 1].
    """

    def __init__(self, dataset_dir: str = "dataset/pendulum"):
        self.files = sorted(Path(dataset_dir).glob("traj_*.npz"))
        assert self.files, f"No .npz files found in {dataset_dir}"

        # Pre-index: (file_idx, frame_idx) for every frame
        first = np.load(self.files[0])
        self.n_frames = first["frames"].shape[0]
        self.index = [
            (fi, t)
            for fi in range(len(self.files))
            for t in range(self.n_frames)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, t = self.index[idx]
        data = np.load(self.files[fi])
        frame = data["frames"][t]                        # (H, W, 3) uint8
        x = torch.from_numpy(frame).permute(2, 0, 1)    # (3, H, W)
        return x.float() / 255.0


class PendulumSeqDataset(Dataset):
    """
    Returns sequences for dynamics model training.
    Each item: tensor (seq_len, 3, H, W) float32 in [0, 1], states (seq_len, S).

    Si seq_len est None, retourne la trajectoire complète.
    Si seq_len est un entier, tire une fenêtre aléatoire de seq_len frames —
    ce qui permet d'entraîner sur de longues trajectoires (500 frames) sans
    charger toute la séquence dans le batch GPU.
    """

    def __init__(self, dataset_dir: str = "dataset/pendulum",
                 seq_len: int | None = None):
        self.files   = sorted(Path(dataset_dir).glob("traj_*.npz"))
        assert self.files, f"No .npz files found in {dataset_dir}"
        self.seq_len = seq_len

        # Inférer la longueur totale depuis le premier fichier
        first = np.load(self.files[0])
        self.traj_len = first["frames"].shape[0]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data   = np.load(self.files[idx])
        frames = data["frames"]   # (T, H, W, 3) uint8
        states = data["states"]   # (T, S) float

        if self.seq_len is not None and self.traj_len > self.seq_len:
            start  = np.random.randint(0, self.traj_len - self.seq_len + 1)
            frames = frames[start:start + self.seq_len]
            states = states[start:start + self.seq_len]

        frames_t = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
        states_t = torch.from_numpy(np.ascontiguousarray(states)).float()
        return frames_t, states_t


def make_seq_dataloaders(
    dataset_dir: str = "dataset/pendulum",
    batch_size: int = 16,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    seq_len: int = 50,
):
    """Dataloaders de séquences (frames, states) pour l'entraînement JEPA.

    seq_len : longueur de la fenêtre tirée aléatoirement dans chaque trajectoire.
    """
    dataset = PendulumSeqDataset(dataset_dir, seq_len=seq_len)
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    pin = torch.cuda.is_available()   # pin_memory non supporté sur MPS
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader


def make_dataloaders(
    dataset_dir: str = "dataset/pendulum",
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
):
    dataset = PendulumFrameDataset(dataset_dir)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader
