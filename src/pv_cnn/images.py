"""Step 2 of the pipeline: from a customer's daily-profile matrix to the image the CNN reads.

Input : X    float32[L, 96]  kWh per 15-minute slot for L calendar days (from data.py)
        days int32[L]        day index of every row (days since config.BASE_DAY), sorted
Output: float32[C + 3, L, 96] image with
        C image channel(s) (see IMAGE_CHANNELS / customer_image):
          'heat'      kW / the customer's 95th percentile, gamma 0.6              -> [0, 1]
          'relsmooth' log2(kW) minus the day's own shoulder level (06-09h, 17-20h), then a
                      15-day x 3-slot median filter on the dense calendar grid, clipped to +-RS_CLIP
                      and scaled                                                  -> [-1, 1]
                      negative = less grid draw at midday than at the shoulders = the PV signature
          'both'      the two above
        + day-of-year sine, day-of-year cosine (season context, repeated over the 96 slots)
        + a mask channel: zeros, except during pretraining where it marks hidden patches
Nothing here is learned; the same function serves pretraining, training, inference and the plots."""
import numpy as np
from scipy.ndimage import median_filter

from config import IMAGE
from data import day_calendar

SHOULDER = np.concatenate([np.arange(a, b) for a, b in IMAGE["shoulder_slots"]])   # default 06-09h and 17-20h
HEAT_GAMMA = float(IMAGE["heat_gamma"])
RS_CLIP = float(IMAGE["clip"])                 # log2 ratio clip before scaling to [-1, 1]
MEDIAN_SIZE = tuple(IMAGE["median_filter"])    # (days, slots)
IMAGE_CHANNELS = {"heat": 1, "relsmooth": 1, "both": 2}


def customer_scale(X):
    """Robust per-customer load level: median over days of the daily mean kWh/slot (used for capacity cues)."""
    s = float(np.median(X.mean(axis=1)))
    floor = 0.05 * float(np.mean(np.abs(X))) + 1e-3
    return max(s, floor)


def heat_pixels(X):
    """linear view: kW / 95th percentile, gamma-stretched low end, in [0, 1]"""
    p95 = max(float(np.percentile(X, 95)), 1e-6)
    return (np.clip(X / p95, 0, 1) ** HEAT_GAMMA).astype(np.float32)


def relsmooth_image(X, days, size=None):
    """log2(kW) minus the day's own shoulder level, median-filtered over (days x slots) on the dense
    calendar grid (missing days filled with 0 before filtering). Returns rows for the given days, unclipped."""
    lk = np.log2(np.maximum(X, 1e-3) * 4.0)                       # kWh / 15 min -> kW, log2
    Z = lk - lk[:, SHOULDER].mean(1, keepdims=True)
    d0 = int(days.min()); grid = np.zeros((int(days.max()) - d0 + 1, X.shape[1]), np.float32)
    grid[days - d0] = Z
    grid = median_filter(grid, size=size or MEDIAN_SIZE)
    return grid[days - d0].astype(np.float32)


def relsmooth_pixels(X, days):
    return (np.clip(relsmooth_image(X, days), -RS_CLIP, RS_CLIP) / RS_CLIP).astype(np.float32)


def customer_image(X, days, mode=None):
    mode = mode or IMAGE["mode"]
    """(C, L, 96) image channels of one customer"""
    if mode == "heat":
        return heat_pixels(X)[None]
    if mode == "relsmooth":
        return relsmooth_pixels(X, days)[None]
    if mode == "both":
        return np.stack([heat_pixels(X), relsmooth_pixels(X, days)])
    raise ValueError(mode)


def calendar_channels(days):
    """(2, L, 96): day-of-year sine and cosine"""
    _, doy = day_calendar(days)
    ang = 2 * np.pi * doy.astype(np.float32) / 365.25
    return np.stack([np.repeat(np.sin(ang)[:, None], 96, 1), np.repeat(np.cos(ang)[:, None], 96, 1)]).astype(np.float32)


def n_channels(mode=None, calendar=None, with_mask=True):
    mode = mode or IMAGE["mode"]; calendar = IMAGE["calendar"] if calendar is None else calendar
    return IMAGE_CHANNELS[mode] + (2 if calendar else 0) + (1 if with_mask else 0)


def build_image(X, days, mode=None, calendar=None, with_mask=True):
    """Full CNN input (n_channels, L, 96): image channel(s) [+ day-of-year sin/cos] [+ zero mask channel].
    calendar=False leaves the network with nothing but the consumption image."""
    mode = mode or IMAGE["mode"]; calendar = IMAGE["calendar"] if calendar is None else calendar
    parts = [customer_image(X, days, mode)]
    if calendar:
        parts.append(calendar_channels(days))
    if with_mask:
        parts.append(np.zeros((1, len(days), X.shape[1]), np.float32))
    return np.concatenate(parts)
