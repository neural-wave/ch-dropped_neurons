"""Local multi-task LSTM baseline for the AIL PV challenge.

Only consumption and calendar information are used. Customer IDs are join keys,
never model inputs. All learned preprocessing is fitted on training customers.
"""
from __future__ import annotations

import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

COLUMNS = ['customer_id', 'energy_consumption_kwh', 'timestamp_local', 'timestamp_epoch']
OUTPUT_COLUMNS = ['customer_id', 'pv_installed', 'pv_installation_date', 'pv_capacity_kwp']
EXT_COLUMNS = ['customer_id', 'extension_date', 'pv_capacity_after_extension_kwp']
TZ = 'Europe/Zurich'


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)


def delimiter(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return csv.Sniffer().sniff(f.read(4096), delimiters=';,\t').delimiter


@dataclass
class CustomerSeries:
    customer_id: str
    dates: pd.DatetimeIndex
    x: np.ndarray
    scale: float
    quality: np.ndarray

def load_series(path, chunksize=250_000):
    """Stream raw measurements into daily hourly profiles without loading the CSV.

    Epoch timestamps determine local clock time. Each observation is interval
    ENERGY: multiply by four for average kW, never by 96. Repeated autumn hours
    are averaged; daily energy and expected coverage retain their 25-hour day.
    Duplicate customer/epoch readings are rejected, including across chunks.
    A compact timestamp bitmap makes validation practical for large files.
    """
    parts, seen, all_ids = [], {}, set()
    invalid_values = 0
    reader = pd.read_csv(path, sep=delimiter(path), encoding='utf-8-sig',
                         dtype={'customer_id': 'string'}, chunksize=chunksize)
    for chunk in reader:
        missing = set(COLUMNS) - set(chunk.columns)
        if missing:
            raise ValueError(f'Missing input columns: {sorted(missing)}')
        if chunk.customer_id.isna().any() or chunk.customer_id.str.len().eq(0).any():
            raise ValueError('Each measurement requires a nonempty customer_id.')
        all_ids.update(chunk.customer_id.astype(str))
        epoch = pd.to_numeric(chunk.timestamp_epoch, errors='raise').to_numpy(float)
        if not np.isfinite(epoch).all() or np.any(epoch % 900 != 0):
            raise ValueError('timestamp_epoch must be integral Unix seconds on a 15-minute grid.')
        steps = epoch.astype(np.int64) // 900
        # Detect arbitrary-order duplicates without retaining millions of Python ints.
        for cid, indices in chunk.groupby('customer_id', sort=False).indices.items():
            values = np.sort(steps[indices])
            if np.any(np.diff(values) == 0):
                raise ValueError('Duplicate customer_id/timestamp_epoch readings.')
            buckets = values // 64
            boundaries = np.r_[0, np.flatnonzero(np.diff(buckets)) + 1, len(values)]
            for lo, hi in zip(boundaries[:-1], boundaries[1:]):
                bits = np.left_shift(np.uint64(1), (values[lo:hi] % 64).astype(np.uint64))
                mask = int(np.bitwise_or.reduce(bits))
                key = (str(cid), int(buckets[lo]))
                old = seen.get(key, 0)
                if old & mask:
                    raise ValueError('Duplicate readings across input chunks.')
                seen[key] = old | mask
        local = pd.to_datetime(epoch, unit='s', utc=True).tz_convert(TZ)
        # Local clock fields are derived from the unambiguous epoch, including DST.
        dates = local.tz_localize(None).normalize()
        kwh = pd.to_numeric(chunk.energy_consumption_kwh, errors='coerce').to_numpy(float)
        valid = np.isfinite(kwh) & (kwh >= 0)
        invalid_values += int((~valid).sum())
        frame = pd.DataFrame({'customer_id': chunk.customer_id.to_numpy(),
                              'date': dates, 'hour': local.hour,
                              'kwh': np.where(valid, kwh, np.nan)})
        parts.append(frame.groupby(['customer_id', 'date', 'hour'], sort=False)
                     .kwh.agg(['sum', 'count']))
    if not parts:
        raise ValueError('The input contains no measurements.')
    hourly = pd.concat(parts).groupby(level=[0, 1, 2], sort=True).sum()
    series = []
    for cid in sorted(all_ids):
        table = hourly.xs(cid, level=0)
        dates = pd.date_range(table.index.get_level_values('date').min(),
                              table.index.get_level_values('date').max(), freq='D')
        index = pd.MultiIndex.from_product([dates, range(24)], names=['date', 'hour'])
        table = table.reindex(index, fill_value=0)
        counts = table['count'].to_numpy(float).reshape(-1, 24)
        energy = table['sum'].to_numpy(float).reshape(-1, 24)
        hourly_kw = np.divide(4 * energy, counts, out=np.full_like(energy, np.nan), where=counts > 0)
        # 92/100 intervals at spring/autumn DST, 96 otherwise.
        midnights = pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=1), freq='D', tz=TZ)
        expected_day = np.diff(midnights.as_unit('ns').asi8) / (900 * 1e9)
        expected = np.full_like(counts, 4.)
        expected[:, 2] = expected_day - 92
        coverage = np.divide(counts, expected, out=np.ones_like(counts), where=expected > 0)
        coverage = np.clip(coverage, 0, 1)
        quality = np.clip(counts.sum(axis=1) / expected_day, 0, 1)
        observed = hourly_kw[np.isfinite(hourly_kw)]
        scale = max(float(np.median(observed)) if len(observed) else 0., 0.01)
        # Per-customer robust scale is computed from that customer's observed input,
        # which is permitted for this retrospective (whole-history) task.
        filled = np.where(np.isfinite(hourly_kw), hourly_kw, scale)
        shape = np.log1p(np.clip(filled / scale, 0, 1e4))
        daily_energy = energy.sum(axis=1)
        energy_feature = np.log1p(daily_energy / (24 * scale))[:, None]
        daytime = filled[:, 9:16].mean(axis=1)
        nighttime = filled[:, [0, 1, 3, 4, 5, 21, 22, 23]].mean(axis=1)
        contrast = np.clip((nighttime - daytime) / scale, -20, 20)[:, None]
        doy = dates.dayofyear.to_numpy() / 365.25 * (2 * np.pi)
        dow = dates.dayofweek.to_numpy() / 7 * (2 * np.pi)
        calendar = np.column_stack([np.sin(doy), np.cos(doy), np.sin(dow), np.cos(dow)])
        elapsed = (np.arange(len(dates)) / 365.25)[:, None]
        x = np.column_stack([shape, coverage, energy_feature, contrast, calendar,
                             elapsed, quality, np.full(len(dates), np.log1p(scale))]).astype('float32')
        series.append(CustomerSeries(str(cid), dates, x, scale, quality.astype('float32')))
    return series, {'customers': len(series), 'invalid_energy_readings_masked': invalid_values,
                    'feature_count': series[0].x.shape[1]}


def read_labels(path):
    labels = pd.read_csv(path, sep=delimiter(path), dtype={'customer_id': 'string'},
                         encoding='utf-8-sig')
    if set(OUTPUT_COLUMNS) - set(labels.columns):
        raise ValueError('The label file is missing required columns.')
    if labels.customer_id.isna().any() or labels.customer_id.duplicated().any():
        raise ValueError('Label IDs must be unique and nonempty.')
    boolean = labels.pv_installed.astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
    if boolean.isna().any():
        raise ValueError('pv_installed must contain TRUE/FALSE.')
    labels['pv_installed'] = boolean.astype(int)
    labels['pv_installation_date'] = pd.to_datetime(labels.pv_installation_date, errors='coerce')
    labels['pv_capacity_kwp'] = pd.to_numeric(labels.pv_capacity_kwp, errors='coerce')
    pv = labels.pv_installed.eq(1)
    if labels.loc[pv, 'pv_installation_date'].isna().any():
        raise ValueError('Every positive label needs a valid commissioning date.')
    cap = labels.loc[pv, 'pv_capacity_kwp'].to_numpy(float)
    if not np.isfinite(cap).all() or np.any(cap <= 0):
        raise ValueError('Every positive label needs a positive finite capacity.')
    return labels.set_index('customer_id')

class FeatureScaler:
    def fit(self, series):
        # Give each customer equal weight, even when histories differ in length.
        means = np.stack([s.x.mean(axis=0) for s in series])
        second = np.stack([(s.x ** 2).mean(axis=0) for s in series])
        self.mean = means.mean(axis=0)
        self.std = np.sqrt(np.maximum(second.mean(axis=0) - self.mean ** 2, 1e-4))
        return self

    def transform(self, x):
        return np.clip((x - self.mean) / self.std, -10, 10).astype('float32')


class CustomerDataset(Dataset):
    def __init__(self, series, scaler, labels=None):
        self.series, self.scaler, self.labels = series, scaler, labels

    def __len__(self):
        return len(self.series)

    def __getitem__(self, i):
        s = self.series[i]
        item = {'x': torch.from_numpy(self.scaler.transform(s.x)), 'series': s}
        if self.labels is not None:
            label = self.labels.loc[s.customer_id]
            pv = float(label.pv_installed)
            offset = (label.pv_installation_date - s.dates[0]).days if pv else 0
            # Out-of-coverage dates are censored, not clipped into false targets.
            item.update(pv=pv, date=float(offset), date_valid=bool(pv and 0 <= offset < len(s.dates)),
                        capacity=float(np.log1p(label.pv_capacity_kwp)) if pv else 0.)
        return item


def collate(items):
    lengths = torch.tensor([len(item['x']) for item in items], dtype=torch.long)
    x = pad_sequence([item['x'] for item in items], batch_first=True)
    mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]
    batch = {'x': x, 'lengths': lengths, 'mask': mask, 'series': [i['series'] for i in items]}
    if 'pv' in items[0]:
        for key in ['pv', 'date', 'date_valid', 'capacity']:
            batch[key] = torch.tensor([i[key] for i in items], dtype=torch.float32)
    return batch

class PVLSTM(nn.Module):
    """Daily recurrent states, customer classification, date localization and kWp.

    Bidirectional context is appropriate because inference observes the complete
    customer history. This model is not an online event detector.
    """
    def __init__(self, input_size, hidden_size=24, dropout=0.20):
        super().__init__()
        self.config = dict(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        width = 2 * hidden_size
        self.dropout = nn.Dropout(dropout)
        self.presence = nn.Sequential(nn.Linear(2 * width, hidden_size), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(hidden_size, 1))
        self.date_head = nn.Sequential(nn.Linear(2 * width, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.active_head = nn.Linear(width, 1)
        self.capacity_head = nn.Sequential(nn.Linear(3 * width, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x, lengths, mask):
        z = self.encoder(x)
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
        h = self.dropout(h)
        mean = (h * mask[..., None]).sum(1) / lengths.to(h.device)[:, None]
        maximum = h.masked_fill(~mask[..., None], -1e4).max(1).values
        pooled = torch.cat([mean, maximum], dim=-1)
        previous = torch.cat([h[:, :1].expand(-1, 7, -1), h], dim=1)[:, :h.shape[1]]
        logits = self.date_head(torch.cat([h, h - previous], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        # A local window after each candidate date helps estimate initial capacity,
        # instead of forcing the regression to describe the final extended system.
        after = nn.functional.avg_pool1d(nn.functional.pad(h.transpose(1, 2), (0, 29)),
                                         kernel_size=30, stride=1).transpose(1, 2)
        event = (logits.softmax(1)[..., None] * after).sum(1)
        capacity = self.capacity_head(torch.cat([pooled, event], dim=-1)).squeeze(-1)
        return {'presence': self.presence(pooled).squeeze(-1), 'date_logits': logits,
                'active_logits': self.active_head(h).squeeze(-1), 'log_capacity': capacity}


def multitask_loss(out, batch, positive_weight):
    presence = nn.functional.binary_cross_entropy_with_logits(
        out['presence'], batch['pv'], pos_weight=positive_weight)
    time = torch.arange(batch['x'].shape[1], device=batch['x'].device)[None, :]
    known_date = batch['date_valid'].bool()
    zero = out['presence'].sum() * 0
    if known_date.any():
        # A seven-day Gaussian target tolerates registration/noisy daily signals.
        target = torch.exp(-0.5 * ((time - batch['date'][:, None]) / 7.) ** 2) * batch['mask']
        target = target / target.sum(1, keepdim=True).clamp_min(1e-12)
        date_loss = -(target * out['date_logits'].log_softmax(1)).sum(1)[known_date].mean()
    else:
        date_loss = zero
    positives = batch['pv'].bool()
    capacity = nn.functional.smooth_l1_loss(out['log_capacity'][positives],
                                           batch['capacity'][positives]) if positives.any() else zero
    active_target = (time >= batch['date'][:, None]).float() * batch['pv'][:, None]
    active_loss = nn.functional.binary_cross_entropy_with_logits(out['active_logits'], active_target, reduction='none')
    # Normalize per customer, so four years of readings is still ONE label.
    active_loss = ((active_loss * batch['mask']).sum(1) / batch['lengths'].to(time.device)).mean()
    return presence + 0.30 * date_loss + 0.60 * capacity + 0.30 * active_loss

def fit_model(series, labels, epochs=60, seed=42, hidden_size=24, batch_size=8, device='cpu', verbose=False):
    seed_everything(seed)
    scaler = FeatureScaler().fit(series)
    dataset = CustomerDataset(series, scaler, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    model = PVLSTM(series[0].x.shape[1], hidden_size).to(device)
    y = labels.loc[[s.customer_id for s in series], 'pv_installed'].to_numpy()
    positive_weight = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), device=device)
    # Capacity bias starts at a train-only geometric prior rather than near zero kWp.
    positive_cap = labels.loc[[s.customer_id for s in series], :].query('pv_installed == 1').pv_capacity_kwp
    if len(positive_cap):
        nn.init.constant_(model.capacity_head[-1].bias, float(np.log1p(positive_cap).mean()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) and k != 'lengths' else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = multitask_loss(model(batch['x'], batch['lengths'], batch['mask']), batch, positive_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
            print(f'Epoch {epoch + 1}/{epochs}: loss={history[-1]:.4f}', flush=True)
    return model.cpu(), scaler, history


@torch.no_grad()
def predict(model, scaler, series, threshold=0.5):
    model.eval()
    rows = []
    loader = DataLoader(CustomerDataset(series, scaler), batch_size=8, collate_fn=collate)
    for batch in loader:
        out = model(batch['x'], batch['lengths'], batch['mask'])
        probabilities = out['presence'].sigmoid().numpy()
        date_prob = out['date_logits'].softmax(1).numpy()
        capacities = np.expm1(np.clip(out['log_capacity'].numpy(), np.log1p(0.01), np.log1p(1e5)))
        for i, s in enumerate(batch['series']):
            # Maximum posterior 30-day neighbourhood matches the challenge's date
            # tolerance better than averaging unrelated peaks across years.
            p = date_prob[i, :len(s.dates)]
            window = np.convolve(p, np.ones(min(15, len(p))), mode='same')
            offset = int(window.argmax())
            lo, hi = max(0, offset - 7), min(len(p), offset + 8)
            local = p[lo:hi]
            offset = int(round(np.dot(np.arange(lo, hi), local) / max(local.sum(), 1e-12)))
            installed = bool(probabilities[i] >= threshold)
            rows.append({'customer_id': s.customer_id, 'pv_installed': installed,
                         'pv_installation_date': s.dates[offset].strftime('%Y-%m-%d') if installed else '',
                         'pv_capacity_kwp': float(capacities[i]) if installed else np.nan,
                         'pv_probability': float(probabilities[i])})
    return pd.DataFrame(rows)


def challenge_metrics(labels, predictions):
    pred = predictions.set_index('customer_id').reindex(labels.index)
    if pred.pv_installed.isna().any():
        raise ValueError('Every labelled customer requires exactly one prediction.')
    y = labels.pv_installed.to_numpy(int)
    yp = pred.pv_installed.astype(int).to_numpy()
    true_positive = (y == 1) & (yp == 1)
    dates = pd.to_datetime(pred.pv_installation_date, errors='coerce')
    errors = (dates - labels.pv_installation_date).dt.days.abs().to_numpy(float)
    cap = pd.to_numeric(pred.pv_capacity_kwp, errors='coerce').to_numpy(float)
    truth = labels.pv_capacity_kwp.to_numpy(float)
    date_scores = np.where(np.isfinite(errors), np.maximum(0, 1 - errors / 30), 0.)
    ape = np.abs(cap - truth) / truth
    cap_scores = np.where(np.isfinite(cap) & (cap > 0) & np.isfinite(ape), np.maximum(0, 1 - ape / .5), 0.)
    f1 = f1_score(y, yp, zero_division=0)
    d = float(date_scores[true_positive].mean()) if true_positive.any() else 0.
    p = float(cap_scores[true_positive].mean()) if true_positive.any() else 0.
    return {'F1': float(f1), 'precision': float(precision_score(y, yp, zero_division=0)),
            'recall': float(recall_score(y, yp, zero_division=0)), 'D': d, 'P': p,
            'main_score': float(f1 * (70 + 15 * d + 15 * p)),
            'true_positives': int(true_positive.sum()),
            'date_MAE_days_on_true_positives': float(errors[true_positive].mean()) if true_positive.any() else None,
            'capacity_MAPE_on_true_positives': float(ape[true_positive].mean()) if true_positive.any() else None}


def cross_validate(series, labels, folds=5, epochs=60, seed=42, hidden_size=24):
    ids = [s.customer_id for s in series]
    if set(ids) != set(labels.index):
        raise ValueError('Development labels and consumption customers must match exactly.')
    y = labels.loc[ids].pv_installed.to_numpy(int)
    if min(np.bincount(y, minlength=2)) < folds:
        raise ValueError('Each class needs at least one customer per fold.')
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof, fold_metrics = [], []
    for fold, (train, valid) in enumerate(splitter.split(ids, y), 1):
        train_series, valid_series = [series[i] for i in train], [series[i] for i in valid]
        # Held-out customer labels never enter fitting, normalization, stopping,
        # or threshold selection. Epoch count and threshold are fixed a priori.
        train_labels = labels.loc[[s.customer_id for s in train_series]]
        model, scaler, _ = fit_model(train_series, train_labels, epochs, seed + fold, hidden_size)
        predictions = predict(model, scaler, valid_series)
        predictions['fold'] = fold
        metrics = challenge_metrics(labels.loc[[s.customer_id for s in valid_series]], predictions)
        fold_metrics.append({'fold': fold, **metrics})
        oof.append(predictions)
        print(f'Fold {fold}/{folds}: F1={metrics["F1"]:.3f}, main score={metrics["main_score"]:.2f}', flush=True)
    return pd.concat(oof, ignore_index=True), pd.DataFrame(fold_metrics)

def save_artifact(model, scaler, path, threshold=0.5, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format_version': 1, 'config': model.config, 'state_dict': model.state_dict(),
                'mean': torch.tensor(scaler.mean), 'std': torch.tensor(scaler.std),
                'threshold': float(threshold), 'metadata': metadata or {}}, path)


def load_artifact(path):
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if artifact['format_version'] != 1:
        raise ValueError('Unsupported model artifact version.')
    model = PVLSTM(**artifact['config'])
    model.load_state_dict(artifact['state_dict'])
    scaler = FeatureScaler()
    scaler.mean, scaler.std = artifact['mean'].numpy(), artifact['std'].numpy()
    return model.eval(), scaler, artifact['threshold']


def write_outputs(predictions, output_dir='.'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if predictions.customer_id.duplicated().any():
        raise ValueError('Predictions must have exactly one row per customer.')
    main = predictions[OUTPUT_COLUMNS].copy()
    positive = main.pv_installed.astype(bool)
    if pd.to_datetime(main.loc[positive, 'pv_installation_date'], errors='coerce').isna().any():
        raise ValueError('PV-positive predictions require valid dates.')
    cap = pd.to_numeric(main.loc[positive, 'pv_capacity_kwp'], errors='coerce').to_numpy(float)
    if not np.isfinite(cap).all() or np.any(cap <= 0):
        raise ValueError('PV-positive predictions require positive finite capacities.')
    main['pv_installed'] = np.where(positive, 'TRUE', 'FALSE')
    main.loc[~positive, ['pv_installation_date', 'pv_capacity_kwp']] = ['', np.nan]
    main.to_csv(output_dir / 'predictions.csv', sep=';', index=False, encoding='utf-8',
                na_rep='', float_format='%.6f')
    # Optional extension detection is deliberately unsupported: one development
    # extension cannot validate a general extension estimator.
    pd.DataFrame(columns=EXT_COLUMNS).to_csv(output_dir / 'pv_extensions_predictions.csv',
                                             sep=';', index=False, encoding='utf-8')
    return output_dir / 'predictions.csv', output_dir / 'pv_extensions_predictions.csv'
