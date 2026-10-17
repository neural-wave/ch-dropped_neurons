"""Step 5: train the CNN pipeline on the human-labelled customers and evaluate it on the labelled customers.

    python train.py --train-labels artifacts/human_labels.csv --pretrained artifacts/pretrained_relsmooth.pt \\
        --tag humans --save-to artifacts/model_humans

Training set : the customers in --train-labels (unlabelled-file customers labelled by humans, build_human_labels.py)
Test set     : ALL customers of the labelled consumption file (paths.labelled_ts) with paths.labels_csv as truth.
               They are never part of the training set (ids present in both are dropped from training).
Writes artifacts/eval_<tag>.json (score, F1, D, P, counts, per-customer table) and prints the per-customer table."""
import argparse, json
import pandas as pd

from config import CACHE_DIR, ARTIFACT_DIR, LABELS_CSV, SEED, IMAGE, TRAIN
from data import load_cache, per_customer, read_labels
from pipeline import CNNPipeline
from scoring import main_score, format_score

pd.set_option("display.width", 200)


def report(name, pred, truth, path):
    r = main_score(pred, truth)
    print(f"\n[{name}] {format_score(r)}")
    m = truth.merge(pred, on="customer_id", suffixes=("_true", "_pred"))
    cols = ["customer_id", "pv_installed_true", "pv_installed_pred", "pv_prob",
            "pv_installation_date_true", "pv_installation_date_pred", "pv_capacity_kwp_true", "pv_capacity_kwp_pred"]
    print(m[cols].to_string(index=False))
    out = {k: v for k, v in r.items() if k != "details"}; out["per_customer"] = m[cols].to_dict(orient="records")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", path)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-labels", nargs="+", required=True, help="label file(s) of unlabelled customers (build_human_labels.py)")
    ap.add_argument("--image", default=IMAGE["mode"], choices=["heat", "relsmooth", "both"])
    ap.add_argument("--calendar", default=IMAGE["calendar"], type=lambda s: s.lower() in ("1", "true", "yes"), help="true|false: day-of-year channels")
    ap.add_argument("--pretrained", default=None, help="conv stack from pretrain.py")
    ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--threshold", type=float, default=TRAIN["threshold"])
    ap.add_argument("--tag", default="humans", help="eval file name: artifacts/eval_<tag>.json")
    ap.add_argument("--save-to", default=None, help="directory for the fitted model")
    args = ap.parse_args()
    ARTIFACT_DIR.mkdir(exist_ok=True)

    truth = read_labels(LABELS_CSV)
    test_data = per_customer(load_cache(CACHE_DIR / "labelled.npz"))
    truth = truth[truth["customer_id"].isin(test_data)].reset_index(drop=True)

    train_data = per_customer(load_cache(CACHE_DIR / "unlabelled.npz"))
    train_labels = pd.concat([read_labels(p) for p in args.train_labels], ignore_index=True).drop_duplicates("customer_id")
    train_labels = train_labels[train_labels["customer_id"].isin(train_data) & ~train_labels["customer_id"].isin(truth["customer_id"])]
    train_labels = train_labels.reset_index(drop=True)
    train_data = {c: train_data[c] for c in train_labels["customer_id"]}
    print(f"train: {len(train_labels)} human-labelled customers ({int(train_labels['pv_installed'].sum())} PV) from {args.train_labels}")
    print(f"test : {len(truth)} labelled customers ({int(truth['pv_installed'].sum())} PV)   image={args.image} calendar={args.calendar}")

    pipe = CNNPipeline(threshold=args.threshold, image=args.image, calendar=bool(args.calendar),
                       pretrained=args.pretrained, epochs=args.epochs, seed=args.seed).fit(train_data, train_labels)
    if args.save_to:
        pipe.save(args.save_to); print("model saved to", args.save_to)
    pred = pipe.predict(test_data)
    report(f"test {len(truth)}", pred, truth, ARTIFACT_DIR / f"eval_{args.tag}.json")


if __name__ == "__main__":
    main()
