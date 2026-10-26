import torch
import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config import SAMPLE_RATE, LOGS_DIR, MODELS
from ctc.model import CTCHead
from ctc.train import load_frozen_upstream


FRAME_HOP_MS = 20.0


def ctc_forced_align(log_probs: torch.Tensor, target_ids: list,
                      blank: int = 0) -> list:
    T, _ = log_probs.shape
    ext  = [blank]
    for c in target_ids:
        ext.append(c); ext.append(blank)
    S      = len(ext)
    NEG_INF = -1e9
    dp      = torch.full((T, S), NEG_INF)
    bp      = torch.zeros((T, S), dtype=torch.long)
    dp[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        dp[0, 1] = log_probs[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            cands = [(dp[t-1, s], s)]
            if s - 1 >= 0:
                cands.append((dp[t-1, s-1], s-1))
            if (s - 2 >= 0 and ext[s] != blank
                    and ext[s] != ext[s-2]):
                cands.append((dp[t-1, s-2], s-2))
            best_val, best_prev = max(cands, key=lambda x: x[0])
            dp[t, s] = best_val + log_probs[t, ext[s]]
            bp[t, s] = best_prev
    end_s = S - 1 if dp[T-1, S-1] >= dp[T-1, S-2] else S - 2
    path  = [end_s]
    for t in range(T - 1, 0, -1):
        end_s = bp[t, end_s].item()
        path.append(end_s)
    path.reverse()
    return [(s - 1) // 2 if s % 2 == 1 else -1 for s in path]


def run_forced_alignment(diag_meta: pd.DataFrame,
                          vocab: dict, model_name: str,
                          device: str = "cpu",
                          ckpt_dir: Path = None,
                          num_layers: int = 13,
                          save: bool = True) -> pd.DataFrame:
    if ckpt_dir is None:
        ckpt_dir = LOGS_DIR / "ctc" / model_name

    upstream, processor = load_frozen_upstream(model_name, device)
    id_to_char = {v: k for k, v in vocab.items()}
    hidden_dim = upstream.config.hidden_size

    token_rows = []
    for layer_idx in range(num_layers):
        ckpt = ckpt_dir / f"layer_{layer_idx}.pt"
        if not ckpt.exists():
            print(f"  Layer {layer_idx}: checkpoint not found, skipping.")
            continue
        head = CTCHead(hidden_dim, len(vocab)).to(device)
        head.load_state_dict(torch.load(ckpt, map_location=device))
        head.eval()

        for file_id, group in diag_meta.groupby("file_id"):
            wav_path = Path(group["wav_path"].iloc[0])
            wav, sr  = sf.read(str(wav_path))
            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            transcript = group["transcript"].iloc[0]
            target_ids = [vocab[c] for c in transcript if c in vocab]
            filtered   = [i for i, c in enumerate(transcript) if c in vocab]
            char_to_t  = {orig: new for new, orig in enumerate(filtered)}

            inputs = processor(wav, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                hs     = upstream(**inputs, output_hidden_states=True).hidden_states
                feats  = hs[layer_idx]
                logits = head(feats)[0]
                lp     = logits.log_softmax(dim=-1).cpu()

            frame_map = ctc_forced_align(lp, target_ids)

            for _, row in group.iterrows():
                orig_idx = int(row["target_char_idx"])
                if orig_idx not in char_to_t:
                    continue
                t_idx  = char_to_t[orig_idx]
                frames = [t for t, idx in enumerate(frame_map) if idx == t_idx]
                if not frames:
                    continue
                token_rows.append({
                    "layer":       layer_idx,
                    "file_id":     file_id,
                    "target_char": row["target_char"],
                    "is_geminate": int(row["is_geminate"]),
                    "duration_ms": round((max(frames) - min(frames) + 1) * FRAME_HOP_MS, 2),
                    "n_frames":    len(frames),
                })

    results = pd.DataFrame(token_rows)
    if save and not results.empty:
        out = ckpt_dir.parent / "forced_alignment_results.csv"
        results.to_csv(out, index=False)
        print(f"Saved -> {out}")
    return results
