"""Central configuration: paths, constants, split settings."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
# folder with the three challenge CSVs: ../challenge_set by default, or PV_DATA_DIR=<folder>
DATA_DIR = Path(os.environ.get("PV_DATA_DIR", PROJECT_DIR.parent / "challenge_set"))

LABELLED_TS = DATA_DIR / "energy_consumption_dev.csv"
UNLABELLED_TS = DATA_DIR / "energy_consumption_unlabelled_dev.csv"
LABELS_CSV = DATA_DIR / "pv_labels_dev.csv"
EXT_LABELS_CSV = DATA_DIR / "pv_extensions_labels_dev.csv"

CACHE_DIR = PROJECT_DIR / "cache"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MODEL_DIR = PROJECT_DIR / "model"          # the shipped model: encoder.pt + downstream.joblib
LABELS_DIR = PROJECT_DIR / "labels"
SPLIT_FILE = PROJECT_DIR / "splits" / "split_32_8.json"

SEED = 42
N_TEST = 8          # labelled customers held out for testing (32 train / 8 test)

SLOTS = 96          # 15-minute intervals per standard day
EMB_DIM = 16        # self-supervised embedding size
TZ = "Europe/Zurich"
BASE_DAY = "2015-01-01"   # day index 0 (safely before any data)
MIN_VALID_SLOTS = 80      # drop partial days with fewer measured slots

# Section 4.1 of the challenge PDF says semicolon separators for the output files
# (the examples in the same PDF use commas). Change here if the organisers clarify.
OUTPUT_SEP = ";"
