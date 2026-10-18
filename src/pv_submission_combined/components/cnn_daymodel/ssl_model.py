"""Self-supervised daily-profile encoder (masked autoencoder on 96-slot days).

Normalisation: each customer's days are divided by that customer's own robust scale
(median daily mean), so the encoder sees *shape* and *relative level* and the same
transform can be applied to an unseen test customer from its own series alone.
"""
import numpy as np
import torch
import torch.nn as nn

from config import SLOTS, EMB_DIM

CLIP_LO, CLIP_HI = -3.0, 8.0


def customer_scale(X):
    """Robust per-customer level: median over days of the daily mean kWh/slot."""
    s = float(np.median(X.mean(axis=1)))
    floor = 0.05 * float(np.mean(np.abs(X))) + 1e-3
    return max(s, floor)


LOG_FLOOR = 0.05      # ratio below which the log-like transform becomes linear (keeps zeros and exports)
MINMAX_PCT = (1, 99)  # per-customer anchors for the 0-1 scaling (use (0, 100) for the literal min and max)


def minmax_anchors(X):
    lo, hi = np.percentile(X, MINMAX_PCT)
    return float(lo), float(max(hi, lo + 1e-6))


def normalize_days(X, scale, mode="linear", days=None):
    """'linear': kW relative to the customer's level, clipped.
    'log'   : asinh of that ratio / LOG_FLOOR, rescaled -> log-like for ratios above ~5%, sign-preserving.
              A 15% midday dip looks the same on a 1 MW load as on a 10 kW load.
    'minmax': per-customer 0-1 scaling between its MINMAX_PCT percentiles (clipped). Every customer then
              spans the same range, whatever its load level.
    'relsmooth': log2(kW / the day's own shoulder level), median-filtered over 15 days x 3 slots (needs days)."""
    if mode == "relsmooth":        # log2 ratio to the day's own shoulders, 15-day x 3-slot median filtered
        from features import relsmooth_image
        return np.clip(relsmooth_image(X, days), -3.0, 3.0).astype(np.float32)
    if mode == "minmax":
        lo, hi = minmax_anchors(X)
        return np.clip((X - lo) / (hi - lo), 0, 1).astype(np.float32)
    r = X / scale
    if mode == "log":
        return (np.arcsinh(r / LOG_FLOOR) / np.arcsinh(1.0 / LOG_FLOOR)).astype(np.float32)   # ratio 1 -> 1.0
    return np.clip(r, CLIP_LO, CLIP_HI).astype(np.float32)


class Encoder(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, norm_mode="linear"):
        super().__init__()
        self.norm_mode = norm_mode
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),   # 48
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(),   # 24
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * (SLOTS // 4), 128), nn.GELU(),
                                  nn.Linear(128, emb_dim))

    def forward(self, x, mask):
        # x: (B, 96) normalised values with masked slots zeroed; mask: (B, 96) 1 where masked
        inp = torch.stack([x, mask], dim=1)
        return self.head(self.conv(inp))


class Decoder(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                 nn.Linear(256, SLOTS))

    def forward(self, z):
        return self.net(z)


def random_block_mask(B, frac=0.3, n_blocks=(1, 3), generator=None):
    """Contiguous-block masks: (B, 96) float tensor with ~frac of slots masked (vectorised)."""
    kmax = n_blocks[1]
    k = torch.randint(n_blocks[0], kmax + 1, (B, 1), generator=generator)          # blocks per sample
    L = (int(frac * SLOTS) // k).clamp(min=1)                                        # block length
    starts = (torch.rand(B, kmax, generator=generator) * (SLOTS - L + 1)).long()     # (B, kmax)
    active = torch.arange(kmax)[None, :] < k                                         # (B, kmax)
    pos = torch.arange(SLOTS)[None, None, :]                                         # (1, 1, 96)
    inside = (pos >= starts[:, :, None]) & (pos < (starts + L)[:, :, None]) & active[:, :, None]
    return inside.any(dim=1).float()


def get_device():
    """CUDA (e.g. the RTX 4090 server) > Apple MPS > CPU. Override with PV_DEVICE=cuda|mps|cpu."""
    import os
    forced = os.environ.get("PV_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def embed(encoder, Xn, batch=4096, device=None):
    """Embed normalised days (n, 96) -> (n, EMB_DIM) float32, no masking."""
    device = device or get_device()
    encoder = encoder.to(device).eval()
    out = []
    for i in range(0, len(Xn), batch):
        x = torch.from_numpy(Xn[i:i + batch]).to(device)
        out.append(encoder(x, torch.zeros_like(x)).cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, EMB_DIM), np.float32)


def save_encoder(encoder, path):
    torch.save({"state_dict": encoder.cpu().state_dict(), "emb_dim": EMB_DIM, "norm_mode": encoder.norm_mode}, path)


def load_encoder(path):
    ck = torch.load(path, map_location="cpu")
    enc = Encoder(ck["emb_dim"], ck.get("norm_mode", "linear")); enc.load_state_dict(ck["state_dict"]); enc.eval()
    return enc
