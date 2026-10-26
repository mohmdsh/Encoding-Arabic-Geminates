import argparse
import pandas as pd
from pathlib import Path
from ast import literal_eval
from config import DATA_DIR
from probing.traditional import run_traditional_probes


def parse_args():
    p = argparse.ArgumentParser(
        description="Traditional acoustic feature baseline probing.")
    p.add_argument("--save-results", action="store_true", default=True)
    return p.parse_args()


def main():
    args  = parse_args()
    aligned_dir = DATA_DIR / "aligned"
    train_df = pd.read_csv(aligned_dir / "aligned_train.csv")
    val_df   = pd.read_csv(aligned_dir / "aligned_val.csv")
    for df in (train_df, val_df):
        df["frame_indices"] = df["frame_indices"].map(literal_eval)
    results = run_traditional_probes(train_df, val_df, save=args.save_results)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
