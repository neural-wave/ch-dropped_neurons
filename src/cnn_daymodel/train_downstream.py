"""Step 3: train the downstream models on the labelled customers and evaluate.

  * fixed stratified split of the 40 labelled customers: 32 train / 8 held-out test
    (saved to splits/split_32_8.json and reused on later runs)
  * fits on the 32, reports the challenge main score on the 8, saves artifacts/
  * --loocv additionally runs leave-one-customer-out over all 40 for a less noisy estimate
  * --train-labels artifacts/human_labels.csv  adds human-labelled unlabelled customers (build_human_labels.py)
    to every training set (32 + humans; each LOOCV fold = 39 + humans)
  * --test-all-labelled  trains on the --train-labels customers only and tests on all 40 ground-truth customers
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CACHE_DIR, ARTIFACT_DIR, LABELS_CSV, SPLIT_FILE, SEED, N_TEST
from data import load_cache, per_customer, read_labels
from ssl_model import load_encoder
from pipeline import PVPipeline
from scoring import main_score, format_score

pd.set_option("display.width", 200)


def get_split(labels):
    if SPLIT_FILE.exists():
        with open(SPLIT_FILE) as f:
            return json.load(f)
    tr, te = train_test_split(labels["customer_id"].tolist(), test_size=N_TEST, random_state=SEED,
                              stratify=labels["pv_installed"].astype(int))
    split = {"train": sorted(tr), "test": sorted(te), "seed": SEED}
    SPLIT_FILE.parent.mkdir(exist_ok=True)
    with open(SPLIT_FILE, "w") as f:
        json.dump(split, f, indent=1)
    return split


def report(name, pred, truth, path=None):
    r = main_score(pred, truth)
    print(f"\n[{name}] {format_score(r)}")
    m = truth.merge(pred, on="customer_id", suffixes=("_true", "_pred"))
    cols = ["customer_id", "pv_installed_true", "pv_installed_pred", "pv_prob",
            "pv_installation_date_true", "pv_installation_date_pred", "pv_capacity_kwp_true", "pv_capacity_kwp_pred"]
    print(m[cols].to_string(index=False))
    if path:
        out = {k: v for k, v in r.items() if k != "details"}
        out["per_customer"] = m[cols].to_dict(orient="records")
        with open(path, "w") as f:
            json.dump(out, f, indent=1, default=str)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loocv", action="store_true", help="also run leave-one-customer-out on all 40")
    ap.add_argument("--encoder", default=str(ARTIFACT_DIR / "encoder.pt"))
    ap.add_argument("--clf", default="gbm", choices=["gbm", "lstm", "hybrid", "linear", "cnn", "ensemble"], help="customer/day classifier family")
    ap.add_argument("--seed", type=int, default=SEED, help="LSTM random seed")
    ap.add_argument("--cnn-pretrained", default=None, help="self-supervised conv weights for --clf cnn (from ssl2d_pretrain.py)")
    ap.add_argument("--image", default="heat", choices=["heat", "relsmooth", "both"], help="--clf cnn input image: linear kW pixels, the relsmooth view, or both")
    ap.add_argument("--date-from", default="gbm", choices=["gbm", "cnn"], help="--clf cnn: day scores used for the commissioning date (gbm day model, or the CNN day head)")
    ap.add_argument("--exclude", nargs="*", default=[], help="customer ids to drop from training and evaluation")
    ap.add_argument("--exclude-train-only", nargs="*", default=[], help="customer ids dropped from training but still evaluated")
    ap.add_argument("--C", type=float, default=None, help="fix the customer logistic-regression C (default: pick by leave-one-out on train)")
    ap.add_argument("--threshold", default="0.5", help="customer decision threshold: a number such as 0.5, or 'tune' (leave-one-out F1 on train; tends to pick a low threshold and add false positives)")
    ap.add_argument("--epochs", type=int, default=150, help="LSTM epochs")
    ap.add_argument("--no-save", action="store_true", help="do not overwrite artifacts/")
    ap.add_argument("--save-to", default=None, help="save the fitted models to this directory instead of artifacts/ (works with --no-save)")
    ap.add_argument("--eval-tag", default="", help="also write eval_*_<tag>.json (works with --no-save)")
    ap.add_argument("--train-labels", nargs="*", default=[], help="extra label files for unlabelled customers (from build_human_labels.py)")
    ap.add_argument("--test-all-labelled", action="store_true", help="train only on --train-labels customers, test on all 40 labelled")
    ap.add_argument("--train-all-labelled", action="store_true",
                    help="final model: train on all 40 labelled customers (+ --train-labels), no held-out test (use --loocv for the estimate)")
    args = ap.parse_args()

    labels = read_labels(LABELS_CSV)
    data = per_customer(load_cache(CACHE_DIR / "labelled.npz"))
    labels = labels[labels["customer_id"].isin(data)].reset_index(drop=True)
    split = get_split(labels)

    extra = pd.DataFrame(columns=labels.columns)
    if args.train_labels:
        unl = per_customer(load_cache(CACHE_DIR / "unlabelled.npz"))
        extra = pd.concat([read_labels(p) for p in args.train_labels], ignore_index=True).drop_duplicates("customer_id")
        extra = extra[extra["customer_id"].isin(unl) & ~extra["customer_id"].isin(labels["customer_id"])].reset_index(drop=True)
        data.update({c: unl[c] for c in extra["customer_id"]}); del unl
        print(f"extra training labels: {len(extra)} unlabelled customers ({int(extra['pv_installed'].sum())} PV) from {args.train_labels}")
    if args.test_all_labelled:
        if not len(extra):
            ap.error("--test-all-labelled needs --train-labels")
        split = {"train": [], "test": sorted(labels["customer_id"]), "seed": SEED}
        if args.loocv:
            print("(--loocv ignored with --test-all-labelled: all 40 labelled customers are already the test set)"); args.loocv = False
    if args.train_all_labelled:
        split = {"train": sorted(labels["customer_id"]), "test": [], "seed": SEED}
    if args.exclude:
        labels = labels[~labels["customer_id"].isin(args.exclude)].reset_index(drop=True)
        split = {k: ([c for c in v if c not in args.exclude] if isinstance(v, list) else v) for k, v in split.items()}
        print(f"excluded {args.exclude}: {len(labels)} customers remain")
    print(f"split: {len(split['train'])} train / {len(split['test'])} test  "
          f"(test PV: {int(labels.set_index('customer_id').loc[split['test'], 'pv_installed'].sum())})")

    sun = None          # no weather / irradiation proxy: consumption and timestamps only
    seq_kwargs = {"epochs": args.epochs, "seed": args.seed}
    if args.clf == "cnn":
        seq_kwargs["image"] = args.image
        if args.cnn_pretrained:
            seq_kwargs["pretrained"] = args.cnn_pretrained
    thr = "tune" if args.threshold == "tune" else float(args.threshold)
    pipe = PVPipeline(load_encoder(args.encoder), sun, clf=args.clf, seq_kwargs=seq_kwargs, threshold=thr, cust_C=args.C, date_from=args.date_from)
    xt = set(args.exclude_train_only)
    if xt:
        split = {k: ([c for c in v if c not in xt] if isinstance(v, list) else v) for k, v in split.items()}
        split["test"] = sorted(set(split["test"]) | xt)
        print(f"excluded from training only: {sorted(xt)} -> {len(split['train'])} train / {len(split['test'])} test")
    print(f"embedding {len(data)} customers ...")
    prep = pipe.prepare(data)

    train_labels = pd.concat([labels[labels["customer_id"].isin(split["train"])], extra], ignore_index=True)
    print(f"fitting on {len(train_labels)} train customers ...")
    pipe.fit(prep, train_labels)
    save_dir = Path(args.save_to) if args.save_to else (None if args.no_save else ARTIFACT_DIR)
    if save_dir is not None:
        pipe.save(save_dir); print("models saved to", save_dir)

    if split["test"]:
        pred_te = pipe.predict({c: prep[c] for c in split["test"]})
        report(f"held-out {len(split['test'])}", pred_te, labels[labels["customer_id"].isin(split["test"])],
               ARTIFACT_DIR / f"eval_holdout_{args.eval_tag}.json" if args.eval_tag else (None if args.no_save else ARTIFACT_DIR / "eval_holdout.json"))

    if args.loocv:
        print("\nleave-one-customer-out over all 40 (each fold refits everything downstream) ...")
        preds = []
        for i, cid in enumerate(labels["customer_id"]):
            p = PVPipeline(pipe.encoder, pipe.sun, clf=args.clf, seq_kwargs=seq_kwargs, threshold=thr, cust_C=args.C, date_from=args.date_from)
            p.fit(prep, pd.concat([labels[(labels["customer_id"] != cid) & ~labels["customer_id"].isin(xt)], extra], ignore_index=True), verbose=False)
            preds.append(p.predict({cid: prep[cid]}))
            print(f"  fold {i + 1}/{len(labels)} done", end="\r", flush=True)
        report("LOOCV 40", pd.concat(preds, ignore_index=True), labels,
               ARTIFACT_DIR / f"eval_loocv_{args.eval_tag}.json" if args.eval_tag else (None if args.no_save else ARTIFACT_DIR / "eval_loocv.json"))


if __name__ == "__main__":
    main()
