import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Audio
from config import SAMPLE_RATE
from ctc.model import build_char_vocab


CTC_TRAIN_SUBSET = 1000
TEXT_COL          = "transcription"


def load_fleurs_arabic(train_subset: int = CTC_TRAIN_SUBSET):
    train_ds = load_dataset("google/fleurs", "ar_eg", split="train")
    val_ds   = load_dataset("google/fleurs", "ar_eg", split="validation")
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    val_ds   = val_ds.cast_column("audio",   Audio(sampling_rate=SAMPLE_RATE))
    if train_subset:
        train_ds = train_ds.select(range(min(train_subset, len(train_ds))))
    vocab = build_char_vocab(train_ds[TEXT_COL])
    return train_ds, val_ds, vocab


class FLEURSArabicCTC(Dataset):
    def __init__(self, hf_dataset, vocab: dict, text_col: str = TEXT_COL):
        self.ds      = hf_dataset
        self.vocab   = vocab
        self.text_col = text_col

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item  = self.ds[idx]
        wav   = torch.tensor(item["audio"]["array"], dtype=torch.float32)
        text  = item[self.text_col]
        label = torch.tensor(
            [self.vocab[c] for c in text if c in self.vocab],
            dtype=torch.long,
        )
        return wav, label, item.get("id", str(idx))


def collate_fn(batch):
    wavs, targets, ids = zip(*batch)
    wav_lengths    = torch.tensor([w.shape[0] for w in wavs])
    target_lengths = torch.tensor([t.shape[0] for t in targets])
    return (
        pad_sequence(wavs,    batch_first=True),
        wav_lengths,
        pad_sequence(targets, batch_first=True),
        target_lengths,
        list(ids),
    )
