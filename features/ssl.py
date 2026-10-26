import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from config import MODELS, FEATS_DIR, SAMPLE_RATE


def load_ssl_model(model_name: str, device: str = "cpu"):
    hf_id = MODELS[model_name]
    processor = Wav2Vec2Processor.from_pretrained(hf_id)
    model     = Wav2Vec2Model.from_pretrained(hf_id)
    model.eval()
    model = model.to(device)
    return model, processor


def extract_all_layers(wav_path: Path, model, processor,
                        out_root: Path, model_name: str,
                        device: str = "cpu") -> int:
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt",
                        padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)

    for layer_i, hidden in enumerate(outputs.hidden_states):
        arr     = hidden.squeeze(0).cpu().numpy()
        out_dir = out_root / model_name / f"layer-{layer_i}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{wav_path.stem}.npy", arr)

    return len(outputs.hidden_states)


def extract_dataset(wav_files: list, model_name: str,
                     feat_root: Path = None, device: str = "cpu"):
    if feat_root is None:
        feat_root = FEATS_DIR
    model, processor = load_ssl_model(model_name, device)
    n_done = n_skip = n_fail = 0
    for wav_path in wav_files:
        layer0 = feat_root / model_name / "layer-0" / f"{wav_path.stem}.npy"
        if layer0.exists():
            n_skip += 1
            continue
        try:
            extract_all_layers(wav_path, model, processor,
                                feat_root, model_name, device)
            n_done += 1
        except Exception as e:
            n_fail += 1
            print(f"  FAIL {wav_path.stem}: {e}")
    print(f"Done: {n_done} extracted, {n_skip} skipped, {n_fail} failed.")
