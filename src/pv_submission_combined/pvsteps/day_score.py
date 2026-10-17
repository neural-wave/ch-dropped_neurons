"""Day-level score P(this day is after a PV installation), from a gradient-boosting classifier.

Inputs: the day table of one customer. Output: the same table with a float32 `score` column in [0, 1].
Fitted on: the 40 labelled customers (days after the true commissioning date are positives); model in state["ail"]["day_model"].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .day_features import DAY_FEATS


def score_days(day_model, tab: pd.DataFrame) -> pd.DataFrame:
    """Add the `score` column (in place) and return the table."""
    tab["score"] = day_model.predict_proba(tab[DAY_FEATS].to_numpy())[:, 1].astype(np.float32)
    return tab
