"""Reproducible training and customer-fold validation of the improved pipeline."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from lstm_pipeline import (PVLSTM, FeatureScaler, CustomerDataset, collate,
                           seed_everything, multitask_loss, predict, challenge_metrics,
                           save_artifact)
from nw26_dropped_neurons.src.simple_lstm_model.pv_refinement import fit_refiner, refine_predictions, save_refiner


def fit_checkpoints(series, labels, epochs=60, seed=42, evaluation_series=None, verbose=False):
    """Keep the 30-epoch date/capacity head and later presence head.

    All epochs and decoder choices are fixed before a fold is fitted. Intermediate
    evaluation uses INPUTS only, never validation labels or early stopping.
    """
    if epochs < 30 or epochs % 30:
        raise ValueError('epochs must be a positive multiple of 30, at least 30.')
    seed_everything(seed)
    scaler = FeatureScaler().fit(series)
    subset = labels.loc[[s.customer_id for s in series]]
    loader = DataLoader(CustomerDataset(series, scaler, subset), batch_size=8, shuffle=True,
                        collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    model = PVLSTM(series[0].x.shape[1], 24)
    y = subset.pv_installed.to_numpy()
    weight = torch.tensor((len(y)-y.sum())/max(y.sum(), 1))
    prior = float(np.log1p(subset.query('pv_installed==1').pv_capacity_kwp).mean())
    nn.init.constant_(model.capacity_head[-1].bias, prior)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    snapshots, predictions, history = {}, {}, []
    evaluation_series = series if evaluation_series is None else evaluation_series
    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(batch['x'], batch['lengths'], batch['mask'])
            loss = multitask_loss(out, batch, weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        if epoch % 30 == 0:
            snapshots[epoch] = copy.deepcopy(model).cpu().eval()
            # A single inference pass per checkpoint keeps this training schedule
            # identical to the measured trajectory experiment.
            predictions[epoch] = predict(model, scaler, evaluation_series,
                                         threshold=0. if epoch == 30 else .5)
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f'Improved fit epoch {epoch}/{epochs}: loss={history[-1]:.4f}', flush=True)
    return snapshots, scaler, predictions, history


def cross_validate_refined(series, labels, epochs=60, folds=5, seed=42):
    ids = [s.customer_id for s in series]
    y = labels.loc[ids].pv_installed.to_numpy(int)
    oof, metrics = [], []
    split = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(split.split(ids, y), 1):
        training = [series[i] for i in tr]
        validation = [series[i] for i in va]
        subset = labels.loc[[s.customer_id for s in training]]
        models, scaler, predictions, _ = fit_checkpoints(training, subset, epochs,
                                                         seed+fold, validation)
        base = predictions[30].copy()
        presence = predictions[epochs]
        if epochs == 30:
            base['pv_installed'] = base.pv_probability >= .5
        else:
            base['pv_installed'] = presence.pv_installed.to_numpy()
            base['pv_probability'] = presence.pv_probability.to_numpy()
        refiner = fit_refiner(training, subset)
        prediction = refine_predictions(validation, base, refiner)
        prediction['fold'] = fold
        score = challenge_metrics(labels.loc[[s.customer_id for s in validation]], prediction)
        oof.append(prediction)
        metrics.append({'fold': fold, **score})
        print(f'Refined fold {fold}/{folds}: F1={score["F1"]:.3f}, score={score["main_score"]:.2f}', flush=True)
    return pd.concat(oof, ignore_index=True), pd.DataFrame(metrics)


def fit_final_refined(series, labels, artifact_dir, epochs=60, seed=42):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    models, scaler, _, history = fit_checkpoints(series, labels, epochs, seed, verbose=True)
    save_artifact(models[30], scaler, artifact_dir/'pv_lstm.pt',
                  metadata={'epochs': 30, 'seed': seed, 'role': 'date_and_capacity'})
    save_artifact(models[epochs], scaler, artifact_dir/'pv_presence.pt',
                  metadata={'epochs': epochs, 'seed': seed, 'role': 'presence'})
    refiner = fit_refiner(series, labels)
    save_refiner(refiner, artifact_dir/'pv_refiner.pt')
    return history

