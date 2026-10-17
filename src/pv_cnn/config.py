"""Loads config.yaml (or the file named by PV_CONFIG) and exposes paths and constants."""
import os
from pathlib import Path
import yaml

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = Path(os.environ.get("PV_CONFIG", PROJECT_DIR / "config.yaml"))
with open(CONFIG_FILE) as _f:
    CFG = yaml.safe_load(_f)


def _path(x):
    p = Path(x)
    return p if p.is_absolute() else (PROJECT_DIR / p).resolve()


DATA_DIR = _path(CFG["paths"]["data_dir"])
LABELLED_TS = DATA_DIR / CFG["paths"]["labelled_ts"]
UNLABELLED_TS = DATA_DIR / CFG["paths"]["unlabelled_ts"]
LABELS_CSV = DATA_DIR / CFG["paths"]["labels_csv"]
CACHE_DIR = _path(CFG["paths"]["cache_dir"])
ARTIFACT_DIR = _path(CFG["paths"]["artifact_dir"])
PLOTS_DIR = _path(CFG["paths"]["plots_dir"])

SEED = int(CFG["seed"])
SLOTS = int(CFG["data"]["slots"])
TZ = CFG["data"]["timezone"]
BASE_DAY = str(CFG["data"]["base_day"])
MIN_VALID_SLOTS = int(CFG["data"]["min_valid_slots"])
ID_PREFIX = str(CFG["data"].get("id_prefix", "") or "")
OUTPUT_SEP = CFG["data"]["output_sep"]

IMAGE, MODEL, TRAIN, PRETRAIN, DATE = CFG["image"], CFG["model"], CFG["train"], CFG["pretrain"], CFG["date"]
