import os
import math
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# torchvision / torchaudio are optional for MOCK mode, required for real data
try:
    from torchvision import transforms, datasets
except ImportError:
    transforms = None
    datasets = None

try:
    import torchaudio
except ImportError:
    torchaudio = None


# ================================================================
# 1. CONFIG
# ================================================================
@dataclass
class Config:
    # high-level
    modality: str = "image"        # "audio", "image", "multimodal"
    use_mock_data: bool = True     # True: synthetic data (runs out-of-box)
                                   # False: expects real files
    image_root: str = "./data/images"   # used if use_mock_data=False
    audio_root: str = "./data/audio"    # used if use_mock_data=False

    # training
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 10
    z_dim: int = 64         # latent dim
    c_dim: int = 256        # context dim
    predict_steps: int = 8  # future steps K
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # negatives
    neg_mode: str = "queue+synthetic"  # "inbatch", "queue", "queue+synthetic"
    queue_size: int = 4096
    tau: float = 0.07   # temperature

    # evaluation
    max_eval_samples: int = 1000
    out_dir: str = "./outputs"


CFG = Config()


# ================================================================
# 2. DATASETS
# ================================================================

class MockAudioDataset(Dataset):
    """Synthetic sine wave dataset with frequency-based labels."""
    def __init__(self, size: int = 2000, seq_len: int = 16000):
        self.size = size
        self.seq_len = seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        freq = np.random.uniform(150, 900)
        x = np.linspace(0, 1, self.seq_len)
        signal = np.sin(2 * np.pi * freq * x) + np.random.normal(0, 0.1, self.seq_len)
        label = 0 if freq < 500 else 1  # low vs high frequency
        wav = torch.from_numpy(signal).float().unsqueeze(0)  # (1, T)
        return {"audio": wav, "label": torch.tensor(label, dtype=torch.long)}


class MockImageDataset(Dataset):
    """Synthetic images: horizontal vs vertical bar patterns."""
    def __init__(self, size: int = 2000, H: int = 64, W: int = 64):
        self.size = size
        self.H = H
        self.W = W

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.zeros(3, self.H, self.W)
        label = np.random.randint(0, 2)
        if label == 0:  # horizontal bars
            for i in range(0, self.H, 8):
                img[:, i:i+4, :] = 1.0
        else:           # vertical bars
            for j in range(0, self.W, 8):
                img[:, :, j:j+4] = 1.0
        img += 0.2 * torch.randn_like(img)
        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}


class MockMultiModalDataset(Dataset):
    """Pair synthetic image + audio with same label."""
    def __init__(self, size: int = 2000, seq_len: int = 16000):
        self.audio_ds = MockAudioDataset(size=size, seq_len=seq_len)
        self.img_ds = MockImageDataset(size=size)

    def __len__(self):
        return len(self.audio_ds)

    def __getitem__(self, idx):
        a = self.audio_ds[idx]
        b = self.img_ds[idx]
        # use audio label as shared label
        return {"audio": a["audio"], "image": b["image"], "label": a["label"]}


class ImageFolderDataset(Dataset):
    """Wrapper around torchvision ImageFolder -> dict."""
    def __init__(self, root: str, image_size: int = 128):
        if transforms is None or datasets is None:
            raise ImportError("torchvision is required for real image dataset")
        self.ds = datasets.ImageFolder(
            root,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}


class AudioFolderDataset(Dataset):
    """
    Expects structure:
        root/
           class1/ *.wav
           class2/ *.wav
    Class is inferred from folder name.
    """
    def __init__(self, root: str, sample_rate: int = 16000, duration: float = 2.0):
        if torchaudio is None:
            raise ImportError("torchaudio is required for real audio dataset")
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.files = []
        self.labels = []
        exts = (".wav", ".flac", ".mp3")
        class_names = sorted(os.listdir(root))
        name_to_idx = {c: i for i, c in enumerate(class_names)}
        for cname in class_names:
            cdir = os.path.join(root, cname)
            if not os.path.isdir(cdir):
                continue
            for fn in os.listdir(cdir):
                if fn.lower().endswith(exts):
                    self.files.append(os.path.join(cdir, fn))
                    self.labels.append(name_to_idx[cname])
        if not self.files:
            raise RuntimeError(f"No audio files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        wav, sr = torchaudio.load(path)  # (C, T)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.shape[1] < self.num_samples:
            pad = self.num_samples - wav.shape[1]
            wav = F.pad(wav, (0, pad))
        else:
            wav = wav[:, :self.num_samples]
        return {"audio": wav, "label": torch.tensor(label, dtype=torch.long)}


class MultiModalFolderDataset(Dataset):
    """
    Pairs an image and audio by index:
      image_root: ImageFolder-style
      audio_root: AudioFolder-style
    """
    def __init__(self, image_root: str, audio_root: str):
        self.img_ds = ImageFolderDataset(image_root)
        self.aud_ds = AudioFolderDataset(audio_root)
        self.length = max(len(self.img_ds), len(self.aud_ds))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.img_ds[idx % len(self.img_ds)]
        aud = self.aud_ds[idx % len(self.aud_ds)]
        # labels may not match; use audio label for probe
        return {"image": img["image"], "audio": aud["audio"], "label": aud["label"]}


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, List[torch.Tensor]] = {}
    for sample in batch:
        for k, v in sample.items():
            if v is None:
                continue
            out.setdefault(k, []).append(v)
    result: Dict[str, torch.Tensor] = {}
    for k, vs in out.items():
        result[k] = torch.stack(vs, dim=0)
    return result


def build_dataloader(cfg: Config) -> DataLoader:
    if cfg.use_mock_data:
        if cfg.modality == "audio":
            ds = MockAudioDataset(size=2000)
        elif cfg.modality == "image":
            ds = MockImageDataset(size=2000)
        elif cfg.modality == "multimodal":
            ds = MockMultiModalDataset(size=2000)
        else:
            raise ValueError("Unknown modality:", cfg.modality)
    else:
        if cfg.modality == "audio":
            ds = AudioFolderDataset(cfg.audio_root)
        elif cfg.modality == "image":
            ds = ImageFolderDataset(cfg.image_root)
        elif cfg.modality == "multimodal":
            ds = MultiModalFolderDataset(cfg.image_root, cfg.audio_root)
        else:
            raise ValueError("Unknown modality:", cfg.modality)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                      collate_fn=collate_batch, drop_last=True)


# ================================================================
# 3. ENCODERS & CONTEXT
# ================================================================

class AudioEncoder(nn.Module):
    """1D conv -> MelSpectrogram -> 2D conv -> (B, T, D)."""
    def __init__(self, z_dim: int, sample_rate: int = 16000):
        super().__init__()
        if torchaudio is None:
            raise ImportError("torchaudio is required for AudioEncoder")
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=64
        )
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, z_dim, 3, 2, 1),
            nn.BatchNorm2d(z_dim),
            nn.ReLU(True),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, 1, T)
        B, C, T = wav.shape
        spec = self.melspec(wav.view(B, T))  # (B, n_mels, T')
        spec = spec.unsqueeze(1)             # (B, 1, n_mels, T')
        feat = self.net(spec)                # (B, D, H, W) ; W ~ time
        B, D, H, W = feat.shape
        feat = feat.mean(dim=2)              # (B, D, W) average over freq
        return feat.transpose(1, 2)          # (B, T, D)


class ImageEncoder(nn.Module):
    """Simple conv net -> spatial grid as sequence (B, T, D)."""
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, z_dim, 3, 1, 1),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B, 3, H, W)
        feat = self.net(img)             # (B, D, H', W')
        B, D, H, W = feat.shape
        feat = feat.view(B, D, H * W)    # (B, D, T)
        return feat.transpose(1, 2)      # (B, T, D)


class ContextGRU(nn.Module):
    def __init__(self, z_dim: int, c_dim: int):
        super().__init__()
        self.gru = nn.GRU(z_dim, c_dim, batch_first=True)
        self.proj = nn.Linear(c_dim, z_dim)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        # z_seq: (B, T, D)
        out, _ = self.gru(z_seq)
        return self.proj(out)  # (B, T, D), context vectors


class CPCMultiModal(nn.Module):
    """
    Shared context model across modalities, separate encoders.
    """
    def __init__(self, z_dim: int, c_dim: int):
        super().__init__()
        self.z_dim = z_dim
        self.encoders = nn.ModuleDict({
            "image": ImageEncoder(z_dim),
            "audio": AudioEncoder(z_dim),
        })
        self.context = ContextGRU(z_dim, c_dim)

    def encode(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        zs: Dict[str, torch.Tensor] = {}
        for mod, encoder in self.encoders.items():
            if mod in batch:
                zs[mod] = encoder(batch[mod])
        return zs

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        zs = self.encode(batch)
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for mod, z_seq in zs.items():
            c_seq = self.context(z_seq)
            out[mod] = {"z": z_seq, "c": c_seq}
        return out


# ================================================================
# 4. NEGATIVE SAMPLER & INFO-NCE LOSS
# ================================================================

class NegativeSampler:
    def __init__(self, mode: str, queue_size: int, z_dim: int, device: torch.device):
        assert mode in ["inbatch", "queue", "queue+synthetic"]
        self.mode = mode
        self.queue_size = queue_size
        self.z_dim = z_dim
        self.device = device

        self.queue = None
        self.queue_ptr = 0
        self.queue_filled = 0
        if "queue" in mode:
            self.queue = torch.zeros(queue_size, z_dim, device=device)

    def _enqueue(self, z: torch.Tensor):
        if self.queue is None:
            return
        B = z.shape[0]
        if B >= self.queue_size:
            self.queue[:] = z[-self.queue_size:].detach()
            self.queue_ptr = 0
            self.queue_filled = self.queue_size
            return
        end = self.queue_ptr + B
        if end <= self.queue_size:
            self.queue[self.queue_ptr:end] = z.detach()
        else:
            first = self.queue_size - self.queue_ptr
            self.queue[self.queue_ptr:] = z[:first].detach()
            self.queue[:B-first] = z[first:].detach()
        self.queue_ptr = (self.queue_ptr + B) % self.queue_size
        self.queue_filled = min(self.queue_filled + B, self.queue_size)

    def compute_loss(
        self,
        c_seq: torch.Tensor,
        z_seq: torch.Tensor,
        tau: float,
        extra_negatives: Optional[torch.Tensor] = None,
        k_steps: int = 1,
    ) -> torch.Tensor:
        """
        c_seq, z_seq: (B, T, D)
        We predict z_{t+1} from c_t (k_steps=1).
        """
        B, T, D = z_seq.shape
        if T < 2:
            return torch.tensor(0.0, device=c_seq.device)

        # use all time steps except last as c_t; next as positive
        z_pos = z_seq[:, 1:, :]   # (B, T-1, D)
        c_t = c_seq[:, :-1, :]    # (B, T-1, D)
        N = B * (T - 1)
        z_pos = z_pos.reshape(N, D)
        c_flat = c_t.reshape(N, D)

        z_all = z_pos

        # add queue negatives
        if self.queue is not None and self.queue_filled > 0:
            z_all = torch.cat([z_all, self.queue[:self.queue_filled]], dim=0)

        # synthetic negatives: gaussian perturbation + shuffled
        if "synthetic" in self.mode:
            noise = 0.15 * torch.randn_like(z_pos)
            z_pert = z_pos + noise
            perm = torch.randperm(N, device=z_pos.device)
            z_shuf = z_pos[perm]
            z_all = torch.cat([z_all, z_pert, z_shuf], dim=0)

        # cross-modal negatives (passed from training loop)
        if extra_negatives is not None:
            z_all = torch.cat([z_all, extra_negatives.detach()], dim=0)

        # logits: (N, M)
        logits = torch.matmul(c_flat, z_all.t()) / tau
        labels = torch.arange(N, device=c_seq.device)
        loss = F.cross_entropy(logits, labels)

        # update queue with current positives
        if self.queue is not None:
            self._enqueue(z_pos)

        return loss


# ================================================================
# 5. TRAINING
# ================================================================

def train(cfg: Config) -> (CPCMultiModal, DataLoader, List[float]):
    loader = build_dataloader(cfg)
    device = torch.device(cfg.device)
    model = CPCMultiModal(cfg.z_dim, cfg.c_dim).to(device)

    # one sampler per modality
    samplers = {
        "image": NegativeSampler(cfg.neg_mode, cfg.queue_size, cfg.z_dim, device),
        "audio": NegativeSampler(cfg.neg_mode, cfg.queue_size, cfg.z_dim, device),
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_history: List[float] = []

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            # move tensors to device
            for k in batch.keys():
                batch[k] = batch[k].to(device)

            outputs = model(batch)  # {mod: {z,c}}
            losses = []

            # compute extra cross-modal negatives for each modality
            for mod, out in outputs.items():
                z = out["z"]    # (B,T,D)
                c = out["c"]    # (B,T,D)
                # build cross-modal negatives from other mods
                extra_neg = []
                for other_mod, out2 in outputs.items():
                    if other_mod == mod:
                        continue
                    z_other = out2["z"].reshape(-1, cfg.z_dim)
                    extra_neg.append(z_other)
                if extra_neg:
                    extra_neg = torch.cat(extra_neg, dim=0)
                else:
                    extra_neg = None

                loss_mod = samplers[mod].compute_loss(
                    c, z, tau=cfg.tau, extra_negatives=extra_neg
                )
                if loss_mod.requires_grad:
                    losses.append(loss_mod)

            if not losses:
                continue

            loss = sum(losses) / len(losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"[Epoch {epoch+1}/{cfg.epochs}] CPC loss: {avg_loss:.4f}")

    # save model weights
    save_path = os.path.join(cfg.out_dir, "cpc_model.pth")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, save_path)
    print("Saved model to:", save_path)
    return model, loader, loss_history


# ================================================================
# 6. REPRESENTATION EVALUATION
# ================================================================

def extract_contexts(
    model: CPCMultiModal,
    loader: DataLoader,
    cfg: Config,
    modality_eval: Optional[str] = None,
):
    device = torch.device(cfg.device)
    model.eval()
    contexts = []
    labels = []

    if modality_eval is None:
        modality_eval = "audio" if cfg.modality != "image" else "image"

    with torch.no_grad():
        for batch in loader:
            for k in batch.keys():
                batch[k] = batch[k].to(device)
            out = model(batch)
            if modality_eval not in out:
                continue
            c_seq = out[modality_eval]["c"]      # (B,T,D)
            c_vec = c_seq[:, -1, :]             # last context
            contexts.append(c_vec.cpu())
            if "label" in batch:
                labels.append(batch["label"].cpu())
            if len(contexts) * cfg.batch_size >= cfg.max_eval_samples:
                break

    if not contexts:
        return None, None, None

    contexts = torch.cat(contexts, dim=0)       # (N,D)
    contexts = F.normalize(contexts, dim=1)
    if labels:
        labels = torch.cat(labels, dim=0).numpy()
    else:
        labels = None
    return contexts.numpy(), labels, modality_eval


def compute_alignment_uniformity(emb: torch.Tensor):
    """
    Wang & Isola (2019)-style alignment and uniformity.
    emb: (N,D) normalized.
    """
    # alignment: distance between x and slightly perturbed version
    noise = torch.randn_like(emb) * 0.1
    emb2 = F.normalize(emb + noise, dim=1)
    align = (emb - emb2).norm(dim=1).pow(2).mean().item()

    # uniformity: log E[exp(-2||x-y||^2)]
    pdist = torch.pdist(emb).pow(2)
    unif = torch.log(torch.exp(-2 * pdist).mean()).item()
    return align, unif


def evaluate_representations(
    model: CPCMultiModal,
    loader: DataLoader,
    cfg: Config,
    loss_history: List[float],
):
    # 1. training curve
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, marker="o")
    plt.title("CPC Training Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("InfoNCE Loss")
    plt.tight_layout()
    curve_path = os.path.join(cfg.out_dir, "training_curve.png")
    plt.savefig(curve_path)
    plt.close()
    print("Saved training curve to:", curve_path)

    # 2. extract contexts
    ctx_np, labels_np, mod_eval = extract_contexts(model, loader, cfg)
    if ctx_np is None:
        print("No contexts extracted; skipping evaluation.")
        return

    emb = torch.from_numpy(ctx_np)
    emb = F.normalize(emb, dim=1)
    align, unif = compute_alignment_uniformity(emb)
    print(f"[{mod_eval}] Alignment (lower better): {align:.4f}")
    print(f"[{mod_eval}] Uniformity (lower better): {unif:.4f}")

    # 3. t-SNE visualization
    print("Running t-SNE (this can take a minute)...")
    # Compatible with older sklearn versions: no n_iter argument
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="random"
    )
    vis = tsne.fit_transform(ctx_np)

    plt.figure(figsize=(6, 6))
    if labels_np is not None:
        scatter = plt.scatter(vis[:, 0], vis[:, 1], c=labels_np, cmap="viridis", alpha=0.7)
        plt.colorbar(scatter, label="Class")
    else:
        plt.scatter(vis[:, 0], vis[:, 1], alpha=0.7)
    plt.title(f"t-SNE of Context Vectors ({mod_eval})")
    plt.tight_layout()
    tsne_path = os.path.join(cfg.out_dir, f"tsne_{mod_eval}.png")
    plt.savefig(tsne_path)
    plt.close()
    print("Saved t-SNE plot to:", tsne_path)

    # 4. Linear probe (if labels exist)
    if labels_np is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            ctx_np, labels_np, test_size=0.3, random_state=42, stratify=labels_np
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(f"Linear probe accuracy on {mod_eval} contexts: {acc*100:.2f}%")
    else:
        print("No labels available -> skipping linear probe.")


# ================================================================
# 7. MAIN + ARGPARSE
# ================================================================
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="CPC multi-modal project runner")
    parser.add_argument(
        "--modality",
        type=str,
        choices=["audio", "image", "multimodal"],
        default=CFG.modality,
        help="Which modality to train on",
    )
    parser.add_argument(
        "--use-mock-data",
        type=str,
        default=str(CFG.use_mock_data),
        help="true/false: use synthetic mock data instead of real files",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=CFG.image_root,
        help="Path to image root (ImageFolder-style) when use-mock-data=false",
    )
    parser.add_argument(
        "--audio-root",
        type=str,
        default=CFG.audio_root,
        help="Path to audio root when use-mock-data=false",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CFG.epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=CFG.out_dir,
        help="Directory to save model and plots",
    )
    return parser.parse_args()


def str2bool(x: str) -> bool:
    return x.lower() in ["1", "true", "yes", "y"]


def main():
    args = parse_args()

    # Override CFG with CLI args
    CFG.modality = args.modality
    CFG.use_mock_data = str2bool(args.use_mock_data)
    CFG.image_root = args.image_root
    CFG.audio_root = args.audio_root
    CFG.epochs = args.epochs
    CFG.out_dir = args.out_dir

    os.makedirs(CFG.out_dir, exist_ok=True)
    print("Using device:", CFG.device)
    print("Modality:", CFG.modality, "| Mock data:", CFG.use_mock_data)
    print("Image root:", CFG.image_root)
    print("Audio root:", CFG.audio_root)
    print("Epochs:", CFG.epochs)
    print("Negative mode:", CFG.neg_mode)
    print("Output dir:", CFG.out_dir)

    model, loader, loss_hist = train(CFG)
    evaluate_representations(model, loader, CFG, loss_hist)


if __name__ == "__main__":
    main()
