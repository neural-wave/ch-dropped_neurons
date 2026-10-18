"""Usage: python test/make_mock.py <out_dir>

Writes a synthetic mock of the organizers' input file and its labels, in the exact schema of the challenge data
(AIL_NeuralWave_Challenge_EN_v1.5.pdf section 3.3 / 3.4), without any real AIL data:

  <out_dir>/test_set.csv                  customer_id;energy_consumption_kwh;timestamp_local;timestamp_epoch
  <out_dir>/mock_labels.csv               customer_id;pv_installed;pv_installation_date;pv_capacity_kwp
  <out_dir>/mock_extensions_labels.csv    customer_id;extension_date;pv_capacity_after_extension_kwp

Conventions copied from the real file: 15-minute intervals, timestamp_local = interval END in Europe/Zurich local time
formatted like 2024-01-01T00:15:00.000, timestamp_epoch = the same instant as Unix seconds, kWh with 6 decimals,
92 / 100 intervals on the DST days, different coverage per customer, ids kept as strings (one has leading zeros).

Customers (all synthetic, deterministic):
  MOCK_NOPV   2022-01-01 .. 2023-12-31, household load only
  MOCK_PV     2022-01-01 .. 2023-12-31, PV from 2022-09-15, 10 kWp
  0042        2022-03-01 .. 2023-12-31, PV from 2022-06-01, 8 kWp, extended to 14 kWp on 2023-05-01 (leading-zero id)
  MOCK_SHORT  2023-01-01 .. 2023-12-31, load only, one year of coverage

The kWp labels are the capacities used to generate the deficit; the model's estimate on synthetic load is not expected
to be accurate, this file exists to test the input/output contract end to end.
"""
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Zurich")
STEP = 900


def household_kwh(local, rng, scale):
    """Base load for one 15-min interval ending at `local` (kWh)."""
    h = local.hour + local.minute / 60
    base = 0.08
    morning = 0.25 * math.exp(-((h - 7.5) ** 2) / 1.5)
    evening = 0.45 * math.exp(-((h - 19.5) ** 2) / 3.0)
    midday = 0.12 * math.exp(-((h - 12.5) ** 2) / 4.0)
    weekend = 1.15 if local.weekday() >= 5 else 1.0
    winter = 1.0 + 0.25 * math.cos(2 * math.pi * (local.timetuple().tm_yday - 15) / 365)
    v = scale * weekend * winter * (base + morning + evening + midday)
    return max(0.0, v * rng.gauss(1.0, 0.18))


def pv_kwh(local, kwp, rng, day_cloud):
    """PV production in the interval ending at `local` (kWh) for a south-facing roof, kWp nominal."""
    h = local.hour + local.minute / 60
    doy = local.timetuple().tm_yday
    season = 0.55 + 0.45 * math.cos(2 * math.pi * (doy - 172) / 365)  # peak at summer solstice
    half_day = 3.0 + 2.0 * season  # hours of light either side of noon
    bell = math.exp(-((h - 13.0) ** 2) / (2 * (half_day / 2.2) ** 2))
    if abs(h - 13.0) > half_day:
        bell = 0.0
    return kwp * 0.25 * 0.85 * season * bell * day_cloud


def gen_customer(cid, start, end, scale, pv_events, rng):
    """Yield rows for one customer. pv_events: list of (date, total_kwp) sorted by date."""
    t = datetime(start.year, start.month, start.day, tzinfo=TZ) + timedelta(seconds=STEP)
    stop = datetime(end.year, end.month, end.day, tzinfo=TZ) + timedelta(days=1)
    cloud, cur_day = 1.0, None
    while t <= stop:
        t_utc = t.astimezone(timezone.utc)
        start_local = (t_utc - timedelta(seconds=STEP)).astimezone(TZ)
        if start_local.date() != cur_day:
            cur_day = start_local.date()
            cloud = min(1.0, max(0.15, rng.gauss(0.75, 0.25)))
        kwp = 0.0
        for d, k in pv_events:
            if start_local.date() >= d:
                kwp = k
        load = household_kwh(start_local, rng, scale)
        gen = pv_kwh(start_local, kwp, rng, cloud) if kwp else 0.0
        kwh = max(0.0, load - gen)
        local_str = (t_utc.astimezone(TZ)).strftime("%Y-%m-%dT%H:%M:%S.000")
        yield f"{cid};{kwh:.6f};{local_str};{int(t_utc.timestamp())}\n"
        t_utc += timedelta(seconds=STEP)
        t = t_utc.astimezone(TZ)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    D = lambda s: datetime.strptime(s, "%Y-%m-%d").date()
    customers = [
        ("MOCK_NOPV", D("2022-01-01"), D("2023-12-31"), 1.0, []),
        ("MOCK_PV", D("2022-01-01"), D("2023-12-31"), 1.3, [(D("2022-09-15"), 10.0)]),
        ("0042", D("2022-03-01"), D("2023-12-31"), 1.1, [(D("2022-06-01"), 8.0), (D("2023-05-01"), 14.0)]),
        ("MOCK_SHORT", D("2023-01-01"), D("2023-12-31"), 0.8, []),
    ]
    n = 0
    with open(out / "test_set.csv", "w", encoding="utf-8", newline="") as f:
        f.write("customer_id;energy_consumption_kwh;timestamp_local;timestamp_epoch\n")
        for cid, s, e, scale, ev in customers:
            for row in gen_customer(cid, s, e, scale, ev, rng):
                f.write(row)
                n += 1
    with open(out / "mock_labels.csv", "w", encoding="utf-8", newline="") as f:
        f.write("customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n")
        for cid, s, e, scale, ev in customers:
            f.write(f"{cid};TRUE;{ev[0][0]};{ev[0][1]:g}\n" if ev else f"{cid};FALSE;;\n")
    with open(out / "mock_extensions_labels.csv", "w", encoding="utf-8", newline="") as f:
        f.write("customer_id;extension_date;pv_capacity_after_extension_kwp\n")
        for cid, s, e, scale, ev in customers:
            for d, k in ev[1:]:
                f.write(f"{cid};{d};{k:g}\n")
    print(f"wrote {out / 'test_set.csv'}: {n} rows, {len(customers)} customers")


if __name__ == "__main__":
    main()
