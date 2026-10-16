"""Focused checks for data semantics, model training, scoring and CLI contract."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lstm_pipeline import *


def fixture(customer, start, end):
    stamps = pd.date_range(start, end, freq='15min', tz=TZ, inclusive='left')
    return pd.DataFrame({'customer_id': customer,
                         'energy_consumption_kwh': np.full(len(stamps), .5),
                         'timestamp_local': stamps.tz_localize(None).astype(str),
                         'timestamp_epoch': stamps.as_unit('ns').asi8 // 10**9})


def run():
    seed_everything()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        spring = fixture('0001', '2024-03-30', '2024-04-02')
        autumn = fixture('0002', '2024-10-26', '2024-10-29')
        frame = pd.concat([spring, autumn], ignore_index=True)
        path = root / 'test.csv'
        frame.sample(frac=1, random_state=42).to_csv(path, sep=';', index=False)
        series, _ = load_series(path, chunksize=17)
        assert [s.customer_id for s in series] == ['0001', '0002']
        assert len(spring) == 96 + 92 + 96 and len(autumn) == 96 + 100 + 96
        for s in series:
            np.testing.assert_allclose(s.quality, 1.)
            np.testing.assert_allclose(s.scale, 2.)  # .5 kWh / .25 h
            assert np.isfinite(s.x).all()
        assert abs(series[0].x[1, 48] - np.log1p(46 / 48)) < 1e-6
        assert abs(series[1].x[1, 48] - np.log1p(50 / 48)) < 1e-6
        # Epoch/ID deduplication must work beyond the chunk boundary.
        pd.concat([frame, frame.iloc[:1]]).to_csv(path, sep=';', index=False)
        try:
            load_series(path, chunksize=17)
        except ValueError as e:
            assert 'Duplicate' in str(e)
        else:
            raise AssertionError('Cross-chunk duplicate accepted')
        # Missing days remain on the sequence; no backward-filling with future data.
        partial = spring.loc[~spring.timestamp_local.str.startswith('2024-03-31')].copy()
        partial.loc[partial.index[0], 'energy_consumption_kwh'] = np.nan
        partial.to_csv(path, sep=',', index=False)
        missing, _ = load_series(path, chunksize=31)
        assert len(missing[0].dates) == 3 and missing[0].quality[1] == 0
        assert np.isfinite(missing[0].x).all()
        labels = pd.DataFrame({'customer_id': ['0001', '0002'], 'pv_installed': [1, 0],
                              'pv_installation_date': [pd.Timestamp('2024-03-31'), pd.NaT],
                              'pv_capacity_kwp': [10., np.nan]}).set_index('customer_id')
        model, scaler, history = fit_model(series, labels, epochs=2, hidden_size=4, batch_size=2)
        assert np.isfinite(history).all()
        predictions = predict(model, scaler, series)
        artifact = root / 'pv.pt'
        save_artifact(model, scaler, artifact)
        restored, norm, threshold = load_artifact(artifact)
        pd.testing.assert_frame_equal(predictions, predict(restored, norm, series, threshold))
        perfect = labels.reset_index().copy()
        assert challenge_metrics(labels, perfect)['main_score'] == 100.
        imperfect = perfect.copy()
        imperfect.loc[0, 'pv_installation_date'] += pd.Timedelta(days=15)
        imperfect.loc[0, 'pv_capacity_kwp'] = 12.5
        assert challenge_metrics(labels, imperfect)['main_score'] == 85.
        invalid = perfect.copy()
        invalid['pv_installation_date'] = pd.NaT
        invalid['pv_capacity_kwp'] = np.nan
        assert challenge_metrics(labels, invalid)['main_score'] == 70.
        perfect['pv_installed'] = perfect.pv_installed.astype(bool)
        write_outputs(perfect, root)
        with (root / 'predictions.csv').open() as f:
            output = list(csv.DictReader(f, delimiter=';'))
        assert output[0]['customer_id'] == '0001'
        assert output[1]['pv_installed'] == 'FALSE'
        assert output[1]['pv_installation_date'] == output[1]['pv_capacity_kwp'] == ''
        assert (root / 'pv_extensions_predictions.csv').read_text().strip() == ';'.join(EXT_COLUMNS)
        # Validate isolated submission with real subprocess inference and a tiny
        # synthetic checkpoint. Never replace the trained deliverable checkpoint.
        import shutil
        submission = root / 'submission'
        submission.mkdir()
        shutil.copy('lstm_pipeline.py', submission / 'lstm_pipeline.py')
        shutil.copy('pv_refinement.py', submission / 'pv_refinement.py')
        shutil.copy('lstm_submission/inference.py', submission / 'inference.py')
        save_artifact(model, scaler, submission / 'artifacts/pv_lstm.pt')
        save_artifact(model, scaler, submission / 'artifacts/pv_presence.pt')
        from nw26_dropped_neurons.src.simple_lstm_model.pv_refinement import fit_refiner, save_refiner
        save_refiner(fit_refiner(series, labels), submission / 'artifacts/pv_refiner.pt')
        frame.to_csv(path, sep=';', index=False)
        result = subprocess.run([sys.executable, str(submission / 'inference.py'), str(path)],
                                cwd=root, text=True, capture_output=True, check=True)
        assert '2 customers' in result.stdout
        assert len(pd.read_csv(root / 'predictions.csv', sep=';')) == 2
    print('PASS: DST, units, chunking, duplicates, missing values/days, IDs, loss, save/load, score, output and CLI.')


if __name__ == '__main__':
    run()
