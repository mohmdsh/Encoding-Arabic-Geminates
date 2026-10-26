import argparse
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from config import MODEL_NAME, MODELS, LOGS_DIR, RESULTS_DIR
from ctc.data import load_fleurs_arabic, FLEURSArabicCTC, collate_fn
from ctc.train import load_frozen_upstream, train_ctc_layer
from ctc.align import run_forced_alignment


def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment 2: Layer-wise CTC training + forced alignment.")
    p.add_argument("--model",       default=MODEL_NAME,
                   choices=list(MODELS.keys()))
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs",      type=int, default=8)
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--train-subset",type=int, default=1000)
    p.add_argument("--diag-csv",    type=str, required=True,
                   help="Path to diagnostic metadata CSV with columns: "
                        "file_id, wav_path, transcript, target_char, "
                        "target_char_idx, is_geminate")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip CTC training, use existing checkpoints.")
    p.add_argument("--save-results", action="store_true", default=True)
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device
    print(f"Model  : {args.model}")
    print(f"Device : {device}")

    if not args.skip_training:
        print("\nLoading FLEURS Arabic ...")
        train_ds, val_ds, vocab = load_fleurs_arabic(args.train_subset)
        train_set = FLEURSArabicCTC(train_ds, vocab)
        val_set   = FLEURSArabicCTC(val_ds,   vocab)
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                   shuffle=True,  collate_fn=collate_fn,
                                   num_workers=2)
        val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                                   shuffle=False, collate_fn=collate_fn,
                                   num_workers=2)
        upstream, processor = load_frozen_upstream(args.model, device)
        num_layers = upstream.config.num_hidden_layers + 1
        ckpt_dir   = LOGS_DIR / "ctc" / args.model
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nTraining {num_layers} CTC heads ...")
        for layer_idx in range(num_layers):
            print(f"\n--- Layer {layer_idx} ---")
            train_ctc_layer(
                layer_idx, args.model, train_loader, val_loader,
                upstream, processor, vocab, device,
                num_epochs=args.epochs, save_dir=ckpt_dir,
            )
        pd.Series(vocab).to_csv(ckpt_dir / "vocab.csv", header=["id"])
        print(f"Checkpoints saved -> {ckpt_dir}")
    else:
        ckpt_dir = LOGS_DIR / "ctc" / args.model
        vocab_csv = ckpt_dir / "vocab.csv"
        if not vocab_csv.exists():
            raise FileNotFoundError(f"vocab.csv not found in {ckpt_dir}. Run training first.")
        vocab_df = pd.read_csv(vocab_csv, index_col=0)
        vocab    = vocab_df["id"].to_dict()
        num_layers = 13

    print("\nRunning forced alignment on diagnostic set ...")
    diag_meta = pd.read_csv(args.diag_csv)
    results   = run_forced_alignment(
        diag_meta, vocab, args.model, device,
        ckpt_dir=ckpt_dir, num_layers=num_layers,
        save=args.save_results,
    )

    summary = (
        results.groupby(["layer", "is_geminate"])["duration_ms"]
        .agg(["mean", "count"])
        .reset_index()
    )
    print("\nDuration by layer and category (ms):")
    print(summary.to_string(index=False))

    if args.save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{args.model}_ctc_duration_summary.csv"
        summary.to_csv(out, index=False)
        print(f"Summary saved -> {out}")


if __name__ == "__main__":
    main()
