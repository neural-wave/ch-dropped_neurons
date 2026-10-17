import os
import numpy as np, pandas as pd, torch, torch.nn as nn, matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm.auto import tqdm

assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then rerun"
dev = "cuda"
print(torch.cuda.get_device_name(0))
torch.backends.cudnn.benchmark = True

D = "./AIL_data_for_neural_wave_v1.1/challenge_set/"
DAYS, SLOTS, Z, EPOCHS, BATCH = 364, 96, 32, 30, 64
sep = lambda f: ";" if open(f).readline().count(";") else ","

def read_energy(f, chunk_rows=2_000_000):
    """Read the CSV in chunks with a row-level progress bar."""
    s = sep(f)
    head = pd.read_csv(f, sep=s, nrows=1000)
    val = [c for c in head.select_dtypes("number").columns if c != "customer_id"][0]
    # estimate total rows from the average line length of the first 1000 lines
    with open(f, "rb") as fh:
        sample = b"".join(fh.readline() for _ in range(1001))
    est_rows = int(os.path.getsize(f) / (len(sample) / 1001))

    chunks = []
    with tqdm(total=est_rows, desc=f"reading {os.path.basename(f)}", unit=" rows", unit_scale=True) as bar:
        for c in pd.read_csv(f, sep=s, usecols=["customer_id", "timestamp_local", val],
                             dtype={val: "float32", "customer_id": "category"}, chunksize=chunk_rows):
            chunks.append(c)
            bar.update(len(c))
    e = pd.concat(chunks, ignore_index=True)
    e["customer_id"] = e.customer_id.astype(str)

    # timestamp parsing in chunks too, so it has a bar
    t = []
    for i in tqdm(range(0, len(e), chunk_rows), desc="parsing timestamps"):
        t.append(pd.to_datetime(e.timestamp_local.iloc[i:i + chunk_rows]))
    e["t"] = pd.concat(t)
    return e.rename(columns={val: "v"})[["customer_id", "t", "v"]]

def to_matrix(s, start):
    """96 (15-min slots, y) x 364 (days, x) image in kW, or None if >10% missing."""
    s = s[(s.t >= start) & (s.t < start + pd.Timedelta(days=DAYS))]
    m = np.full((SLOTS, DAYS), np.nan, np.float32)
    m[(s.t.dt.hour * 4 + s.t.dt.minute // 15).values, (s.t.dt.normalize() - start).dt.days.values] = s.v.values * 4
    return None if np.isnan(m).mean() > 0.1 else np.nan_to_num(m)

# ---- unlabelled: every non-overlapping 364-day window of every customer is one image
e = read_energy(D + "energy_consumption_unlabelled_dev.csv")
X = []
groups = e.groupby("customer_id", sort=False)
for cid, s in tqdm(groups, total=groups.ngroups, desc="unlabelled images"):
    start = s.t.min().normalize()
    while start + pd.Timedelta(days=DAYS) <= s.t.max():
        m = to_matrix(s, start)
        if m is not None:
            X.append(m)
        start += pd.Timedelta(days=DAYS)
del e, groups
X = np.log1p(np.clip(np.stack(X), 0, None))
SCALE = np.percentile(X, 99.9)  # global scale: keeps the size of the site in the image
X = torch.from_numpy(X / SCALE).float()[:, None]
print(f"{len(X)} training images, shape {tuple(X.shape[1:])}")

# ---- labelled PV customers: the 364 days after installation (or the last 364 days)
e = read_energy(D + "energy_consumption_dev.csv")
lab = pd.read_csv(D + "pv_labels_dev.csv", sep=sep(D + "pv_labels_dev.csv"))
lab = lab[lab.pv_capacity_kwp.notna()]
lab["d"] = pd.to_datetime(lab.pv_installation_date)
groups = dict(tuple(e.groupby("customer_id")))
Xl, kwp, ids = [], [], []
for _, r in tqdm(lab.iterrows(), total=len(lab), desc="labelled images"):
    s = groups.get(r.customer_id)
    if s is None:
        continue
    start = r.d.normalize()
    if pd.isna(start) or start + pd.Timedelta(days=DAYS) > s.t.max():
        start = (s.t.max() - pd.Timedelta(days=DAYS)).normalize() + pd.Timedelta(days=1)
    m = to_matrix(s, start)
    if m is not None:
        Xl.append(m); kwp.append(r.pv_capacity_kwp); ids.append(r.customer_id)
del e, groups
Xl = torch.from_numpy(np.log1p(np.clip(np.stack(Xl), 0, None)) / SCALE).float()[:, None]
kwp = np.array(kwp)
print(f"{len(Xl)} labelled images")

# ---- convolutional autoencoder
class AE(nn.Module):
    def __init__(self, z=Z):
        super().__init__()
        self.enc = nn.Sequential(                                   # 96x364
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),                   # 48x182
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),                  # 24x91
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),                  # 12x46
            nn.Flatten(), nn.Linear(64 * 12 * 46, z))
        self.dec = nn.Sequential(
            nn.Linear(z, 64 * 12 * 46), nn.ReLU(), nn.Unflatten(1, (64, 12, 46)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),         # 24x92
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),         # 48x184
            nn.ConvTranspose2d(16, 1, 4, 2, 1))                     # 96x368
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)[..., :DAYS], z

model = AE().to(dev)
opt = torch.optim.Adam(model.parameters(), 1e-3)
scaler = torch.amp.GradScaler("cuda")
perm = torch.randperm(len(X)); n_val = max(1, len(X) // 10)
tr, va = X[perm[n_val:]].to(dev), X[perm[:n_val]].to(dev)   # whole dataset on the GPU

epochs = tqdm(range(EPOCHS), desc="epochs")
for ep in epochs:
    model.train()
    order = torch.randperm(len(tr), device=dev)
    batches = tqdm(range(0, len(tr), BATCH), desc=f"epoch {ep + 1}/{EPOCHS}", leave=False)
    for i in batches:
        xb = tr[order[i:i + BATCH]]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = nn.functional.mse_loss(model(xb)[0].float(), xb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        batches.set_postfix(loss=f"{loss.item():.4f}")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        val_loss = np.mean([nn.functional.mse_loss(model(va[i:i + 256])[0].float(), va[i:i + 256]).item()
                            for i in range(0, len(va), 256)])
    epochs.set_postfix(train=f"{loss.item():.4f}", val=f"{val_loss:.4f}")

# ---- encode the labelled customers and correlate every latent feature with kWp
with torch.no_grad():
    feats = model.enc(Xl.to(dev)).cpu().numpy()
F = pd.DataFrame(feats, columns=[f"z{j}" for j in range(Z)])
F["log_mean_kw"] = np.log(np.expm1(Xl.numpy() * SCALE).mean(axis=(1, 2, 3)))  # baseline: site size
table = pd.DataFrame({
    "spearman": [spearmanr(F[c], kwp)[0] for c in tqdm(F.columns, desc="correlations", leave=False)],
    "pearson_log_kwp": [np.corrcoef(F[c], np.log(kwp))[0, 1] for c in F],
}, index=F.columns).sort_values("spearman", key=abs, ascending=False)

# chance level: best |Spearman| among Z random features when kWp is shuffled
null = [np.max(np.abs([spearmanr(F[c], np.random.permutation(kwp))[0] for c in F.columns[:Z]]))
        for _ in tqdm(range(200), desc="chance level (shuffles)")]
print(f"\n|Spearman| reachable by chance with {Z} features and n={len(kwp)}: {np.percentile(null, 95):.2f}\n")
print(table.round(2).head(10).to_string())

# ---- plots: one image, its reconstruction, best feature vs kWp
best = table.index[0]
with torch.no_grad():
    rec = model(Xl[:1].to(dev))[0].cpu().numpy()[0, 0]
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for a, img, t in [(ax[0], Xl[0, 0].numpy(), f"{ids[0]} input"), (ax[1], rec, "reconstruction")]:
    a.imshow(img, aspect="auto", cmap="viridis", origin="lower")
    a.set(title=t, xlabel="day", ylabel="15-min slot")
ax[2].scatter(F[best], kwp); ax[2].set(xlabel=best, ylabel="kWp", yscale="log", title=f"best feature: {best}")
plt.tight_layout(); plt.show()