import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from config import MODELS, LOGS_DIR, SAMPLE_RATE
from ctc.model import CTCHead


def load_frozen_upstream(model_name: str, device: str):
    hf_id     = MODELS[model_name]
    processor = Wav2Vec2Processor.from_pretrained(hf_id)
    upstream  = Wav2Vec2Model.from_pretrained(hf_id, output_hidden_states=True)
    upstream.eval()
    for p in upstream.parameters():
        p.requires_grad = False
    upstream = upstream.to(device)
    return upstream, processor


def get_hidden_states(upstream, processor, wavs_padded,
                       wav_lengths, device: str):
    inputs = processor(
        [w[:l].numpy() for w, l in zip(wavs_padded, wav_lengths)],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = upstream(**inputs)
    frame_lengths = torch.IntTensor(
        [min(l.item() // 320, outputs.hidden_states[0].size(1))
         for l in wav_lengths]
    ).to(device)
    return outputs.hidden_states, frame_lengths


def train_ctc_layer(layer_idx: int, model_name: str,
                     train_loader, val_loader,
                     upstream, processor,
                     vocab: dict, device: str,
                     num_epochs: int = 8, lr: float = 1e-4,
                     save_dir: Path = None) -> dict:
    if save_dir is None:
        save_dir = LOGS_DIR / "ctc" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    hidden_dim = upstream.config.hidden_size
    vocab_size = len(vocab)
    head       = CTCHead(hidden_dim, vocab_size).to(device)
    optimizer  = torch.optim.Adam(head.parameters(), lr=lr)
    ctc_loss   = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    history = defaultdict(list)

    for epoch in range(num_epochs):
        head.train()
        epoch_loss = 0.0
        for wavs_padded, wav_lengths, targets_padded, target_lengths, _ in tqdm(
                train_loader, desc=f"Layer {layer_idx} Epoch {epoch+1}"):
            wavs_padded    = wavs_padded.to(device)
            targets_padded = targets_padded.to(device)
            hidden_states, frame_lengths = get_hidden_states(
                upstream, processor, wavs_padded, wav_lengths, device)
            feats   = hidden_states[layer_idx]
            logits  = head(feats)
            log_p   = logits.log_softmax(dim=-1).permute(1, 0, 2)
            loss    = ctc_loss(log_p, targets_padded,
                               frame_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        history["train_loss"].append(epoch_loss / len(train_loader))

    torch.save(head.state_dict(), save_dir / f"layer_{layer_idx}.pt")
    return dict(history)
