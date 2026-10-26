import argparse
import torch
import pandas as pd
from pathlib import Path
from ast import literal_eval
from config import (
    MODEL_NAME, FEATS_DIR, LOGS_DIR, DATA_DIR, RESULTS_DIR,
    WAV_DIR, TEST_WAV_DIR, TG_DIR, TEST_TG_DIR, MODELS, CLASS_GROUPS,
)
from corpus import build_train_val_split, load_consonants_from_dir
from features.ssl import extract_dataset
from probing.align import align_consonants
from probing.probes import run_pair_probes


def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment 1: Layer-wise probing of SSL representations.")
    p.add_argument("--model",  default=MODEL_NAME,
                   choices=list(MODELS.keys()))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-extraction", action="store_true",
                   help="Skip SSL feature extraction (use cached .npy files).")
    p.add_argument("--save-results", action="store_true", default=True,
                   help="Save per-pair CSV results to results/.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Model  : {args.model}")
    print(f"Device : {args.device}")

    train_files, val_files, train_ids, val_ids = build_train_val_split(
        WAV_DIR, TG_DIR)
    print(f"Train : {len(train_files)} files | Val : {len(val_files)} files")

    if not args.skip_extraction:
        print("\nExtracting SSL features ...")
        extract_dataset(train_files + val_files, args.model,
                         feat_root=FEATS_DIR, device=args.device)

    print("\nLoading consonants ...")
    train_cons = load_consonants_from_dir(TG_DIR, train_ids, label="train")
    val_cons   = load_consonants_from_dir(TEST_TG_DIR, label="val")

    print("\nAligning timestamps to frames ...")
    aligned_dir = DATA_DIR / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    train_aligned = align_consonants(train_cons, WAV_DIR)
    val_aligned   = align_consonants(val_cons,   TEST_WAV_DIR)
    train_aligned.to_csv(aligned_dir / "aligned_train.csv", index=False)
    val_aligned.to_csv(aligned_dir   / "aligned_val.csv",   index=False)

    layer0_dir = FEATS_DIR / args.model / "layer-0"
    extracted  = {p.stem for p in layer0_dir.glob("*.npy")} if layer0_dir.exists() else set()
    train_aligned = train_aligned[train_aligned["file_id"].isin(extracted)].copy()
    val_aligned   = val_aligned[val_aligned["file_id"].isin(extracted)].copy()
    for df in (train_aligned, val_aligned):
        if df["frame_indices"].dtype == object and isinstance(
                df["frame_indices"].iloc[0], str):
            df["frame_indices"] = df["frame_indices"].map(literal_eval)

    num_layers = 12
    feat_root  = FEATS_DIR / args.model
    all_results = {}

    for cls_name, pairs in CLASS_GROUPS.items():
        print(f"\n=== {cls_name} ===")
        log_base = LOGS_DIR / cls_name.replace(" ","_").replace("(","").replace(")","")
        for sing, gem in pairs:
            n_tr = (train_aligned["phone"] == gem).sum()
            n_va = (val_aligned["phone"]   == gem).sum()
            if n_tr < 5 or n_va < 2:
                print(f"  {sing}/{gem}: SKIP (too few geminates)")
                continue
            print(f"  {sing}/{gem} ...")
            res = run_pair_probes(sing, gem, train_aligned, val_aligned,
                                   feat_root, log_base, num_layers, args.model)
            all_results[(sing, gem)] = res

    if args.save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        rows = []
        for cls_name, pairs in CLASS_GROUPS.items():
            for sing, gem in pairs:
                if (sing, gem) not in all_results:
                    continue
                for layer_i, metrics in all_results[(sing, gem)].items():
                    row = {
                        "model": args.model, "class": cls_name,
                        "singleton": sing, "geminate": gem,
                        "pair": f"{sing}/{gem}", "layer": layer_i,
                    }
                    cls = metrics.get("cls", {})
                    row["bal_acc"]      = cls.get("balanced_acc", float("nan"))
                    row["f1_gem"]       = cls.get("f1_gem",       float("nan"))
                    row["f1_gem_ci_lo"] = cls.get("ci_gem", (float("nan"),))[0]
                    row["f1_gem_ci_hi"] = cls.get("ci_gem", (float("nan"),float("nan")))[1]
                    row["f1_sing"]      = cls.get("f1_sing",      float("nan"))
                    row["rand_f1_gem"]  = cls.get("rand_f1_gem",  float("nan"))
                    dg = metrics.get("dur_gem", {})
                    row["r2_gem"]       = dg.get("r2",  float("nan"))
                    row["r2_gem_ci_lo"] = dg.get("ci", (float("nan"),))[0]
                    row["r2_gem_ci_hi"] = dg.get("ci", (float("nan"),float("nan")))[1]
                    row["rand_r2_gem"]  = dg.get("rand_r2", float("nan"))
                    rows.append(row)
        out = RESULTS_DIR / f"{args.model}_probe_results.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nResults saved -> {out}")


if __name__ == "__main__":
    main()
