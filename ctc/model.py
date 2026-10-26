import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CTCHead(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x):
        return self.net(x)


class BLSTMDecoder(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int,
                 num_layers: int = 3, hidden_dim: int = 1024,
                 dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        current = input_dim
        out_dim = hidden_dim * 2
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "rnn":  nn.LSTM(current, hidden_dim, bidirectional=True,
                                num_layers=1, batch_first=True),
                "norm": nn.LayerNorm(out_dim),
                "drop": nn.Dropout(p=dropout),
                "proj": nn.Linear(out_dim, out_dim),
            }))
            current = out_dim
        self.linear = nn.Linear(current, vocab_size)

    def _layer_fwd(self, d, x, x_len):
        if not self.training:
            d["rnn"].flatten_parameters()
        packed     = pack_padded_sequence(x, x_len.cpu(), batch_first=True,
                                          enforce_sorted=False)
        out, _     = d["rnn"](packed)
        out, _     = pad_packed_sequence(out, batch_first=True)
        out        = d["norm"](out)
        out        = d["drop"](out)
        out        = d["proj"](out)
        return out

    def forward(self, x, x_len):
        for layer in self.layers:
            x = self._layer_fwd(layer, x, x_len)
        return self.linear(x)

    def greedy_decode(self, logits, id_to_char: dict) -> list:
        preds = logits.argmax(dim=-1).cpu().tolist()
        results = []
        for seq in preds:
            chars, prev = [], -1
            for p in seq:
                if p != 0 and p != prev:
                    chars.append(id_to_char.get(p, ""))
                prev = p
            results.append("".join(chars))
        return results


def build_char_vocab(texts) -> dict:
    chars = sorted({c for t in texts for c in t if c.strip()})
    vocab = {"<blank>": 0}
    for i, c in enumerate(chars, 1):
        vocab[c] = i
    return vocab
