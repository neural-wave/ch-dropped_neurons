"""One-shot environment check for the GPU server:  python check_env.py
Prints python, GPU, library versions and the data/artifact files, and exits 1 if anything needed is missing."""
import importlib, platform, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import LABELLED_TS, UNLABELLED_TS, LABELS_CSV  # noqa: E402
ok, missing = [], []

print(f"python {platform.python_version()}  ({sys.executable})   {platform.platform()}")
if shutil.which("nvidia-smi"):
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True)
    print("nvidia-smi:", r.stdout.strip() or r.stderr.strip())
else:
    print("nvidia-smi: not found")

print("\nlibraries (requirements.txt):")
for mod, pip_name in [("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"),
                      ("torch", "torch"), ("joblib", "joblib")]:
    try:
        m = importlib.import_module(mod)
        print(f"  OK       {pip_name:14s} {getattr(m, '__version__', '?')}"); ok.append(pip_name)
    except Exception as e:
        print(f"  MISSING  {pip_name:14s} ({e.__class__.__name__})"); missing.append(pip_name)

if "torch" in ok:
    import torch
    cuda = torch.cuda.is_available()
    print(f"\ntorch cuda available: {cuda}   torch cuda build: {torch.version.cuda}")
    if cuda:
        dev = torch.device("cuda")
        print(f"  gpu: {torch.cuda.get_device_name(0)}   "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
        x = torch.randn(2048, 2048, device=dev); y = (x @ x).sum().item()
        print(f"  cuda matmul test: OK ({y:.3g})")
    elif shutil.which("nvidia-smi"):
        missing.append("torch-with-CUDA (an NVIDIA GPU is present but torch cannot see it: CPU-only build or driver mismatch)")
    else:
        print("  no NVIDIA GPU on this machine -> will fall back to", "mps" if torch.backends.mps.is_available() else "cpu")

print("\nfiles:")
def show(p, needed):
    exists = p.exists()
    size = f"{p.stat().st_size / 2**20:,.0f} MB" if exists and p.is_file() else ""
    print(f"  {'OK     ' if exists else 'MISSING'}  {p}  {size}")
    if needed and not exists:
        missing.append(str(p.name))
show(LABELLED_TS, True)
show(UNLABELLED_TS, True)
show(LABELS_CSV, True)
show(HERE / "labels" / "human_labels.csv", True)
show(HERE / "cache" / "labelled.npz", False)
show(HERE / "cache" / "unlabelled.npz", False)
show(HERE / "artifacts" / "encoder.pt", False)
show(HERE / "artifacts" / "heatcnn_ssl_relsmooth.pt", False)
show(HERE / "model" / "encoder.pt", False)
show(HERE / "model" / "downstream.joblib", False)
du = shutil.disk_usage(HERE)
print(f"\ndisk free here: {du.free / 2**30:.0f} GiB")

if missing:
    print("\nRESULT: MISSING ->", ", ".join(missing))
    print("fix:   bash setup_server.sh (creates .venv + requirements); data: PV_DATA_DIR=<folder with the challenge CSVs>")
    sys.exit(1)
print("\nRESULT: OK - everything needed is present")
