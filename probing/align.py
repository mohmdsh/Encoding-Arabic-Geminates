import math
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def _cnn_output_length(L: int, k: int, s: int) -> int:
    return math.floor((L - k) / s) + 1


def wav2vec2_frame_count(n_samples: int) -> int:
    L = n_samples
    for k, s in [(10,5),(3,2),(3,2),(3,2),(3,2),(2,2),(2,2)]:
        L = _cnn_output_length(L, k, s)
    return L


def time_to_frame(t_sec: float, step_s: float = 0.02) -> int:
    return int(t_sec / step_s)


def align_consonants(consonants_df: pd.DataFrame,
                      wav_dir: Path) -> pd.DataFrame:
    records = []
    skipped = 0
    for file_id, group in tqdm(consonants_df.groupby("file_id"),
                                desc="Aligning", unit="file"):
        wav_path = wav_dir / f"{file_id}.wav"
        if not wav_path.exists():
            skipped += 1
            continue
        try:
            info     = sf.info(str(wav_path))
            n_frames = wav2vec2_frame_count(info.frames)
        except Exception as e:
            print(f"  SKIP {file_id}: {e}")
            skipped += 1
            continue
        for _, row in group.iterrows():
            start_idx  = time_to_frame(row["start"])
            end_idx    = time_to_frame(row["end"])
            frame_idxs = list(range(max(0, start_idx),
                                    min(n_frames, end_idx + 1)))
            rec = row.to_dict()
            rec["frame_indices"] = frame_idxs
            records.append(rec)
    if skipped:
        print(f"  Skipped {skipped} files (wav not found).")
    return pd.DataFrame(records)
