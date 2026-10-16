"""Offline challenge entry point: python inference.py path/to/test_set.csv."""
import argparse
from pathlib import Path
from lstm_pipeline import load_series, seed_everything, write_outputs
from nw26_dropped_neurons.src.simple_lstm_model.pv_refinement import predict_bundle


def main():
    parser = argparse.ArgumentParser(description='Predict initial PV presence, date and capacity.')
    parser.add_argument('test_csv', type=Path)
    args = parser.parse_args()
    seed_everything()
    artifact_dir = Path(__file__).resolve().parent / 'artifacts'
    required = ['pv_lstm.pt', 'pv_presence.pt', 'pv_refiner.pt']
    missing = [name for name in required if not (artifact_dir/name).exists()]
    if missing:
        parser.error('Missing model artifacts: ' + ', '.join(missing) + '. Run the improved notebook export.')
    series, _ = load_series(args.test_csv)
    predictions = predict_bundle(artifact_dir, series)
    if set(predictions.customer_id) != {s.customer_id for s in series}:
        raise RuntimeError('Output customer IDs do not match the input.')
    write_outputs(predictions)
    print(f'Wrote both submission files for {len(predictions)} customers.')


if __name__ == '__main__':
    main()
