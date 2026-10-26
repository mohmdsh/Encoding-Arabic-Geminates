# Arabic Geminate Probing — 

Replication code for the probing analysis in:

> AlShakhori & Issa — *Encoding of Arabic Geminate Consonants in Self-Supervised Speech Representations: A Probing Analysis*

## Repository Structure

```
github/
├── config.py                # paths, models, phoneme groups, hyperparameters
├── corpus.py                # TextGrid parsing, Buckwalter decoding, train/val split
├── features/
│   ├── acoustic.py          # MFCC, Fbank, MelSpec feature extraction
│   └── ssl.py               # SSL hidden-state extraction (all layers)
├── probing/
│   ├── align.py             # timestamp → frame-index alignment
│   ├── probes.py            # logistic regression (F1) + linear regression (R²) probes
│   └── traditional.py       # traditional feature baseline probes
├── run_experiment1.py       # Experiment 1: layer-wise SSL probing
└── run_traditional.py       # Traditional feature baseline
```

## Setup

```bash
pip install -r requirements.txt
```

Place the [Arabic Speech Corpus](https://www.arabicspeechcorpus.com) under `arabic-speech-corpus/`.

## Corpus Layout Expected

```
arabic-speech-corpus/
├── wav/                 # training WAV files
├── textgrids/           # training TextGrid files
└── test set/
    ├── wav/
    └── textgrids/
```

## Experiment 1 — Layer-wise Probing

```bash
python run_experiment1.py --model wav2vec2-base --device cuda
```

Options:
- `--model`            : `wav2vec2-base` | `hubert-base` | `wavlm-base`
- `--device`           : `cuda` | `cpu`
- `--skip-extraction`  : skip SSL feature extraction (use cached `.npy` files)

Results are saved to `data/arabic-geminate/results/<model>_probe_results.csv`.

## Traditional Baselines

```bash
python run_traditional.py
```

Requires `data/arabic-geminate/aligned/` from Experiment 1.
Results saved to `data/arabic-geminate/results/traditional_probe_results.csv`.

## Output Format

| Column | Description |
|---|---|
| `model` | SSL model name |
| `class` | Phoneme manner class |
| `pair` | singleton/geminate pair (e.g. `t/tt`) |
| `layer` | Transformer layer index (0 = CNN output) |
| `f1_gem` | Geminate classification F1 |
| `r2_gem` | Geminate duration regression R² |
| `f1_gem_ci_lo/hi` | 95% bootstrap CI bounds |

## Experiment 2 — CTC Training + Forced Alignment

### Step 1 — Prepare diagnostic metadata CSV

Create a CSV with one row per target token in your minimal-pair set:

| Column | Description |
|---|---|
| `file_id` | WAV file stem |
| `wav_path` | Absolute path to WAV file |
| `transcript` | Full utterance transcript (no diacritics, geminate marked with shadda character) |
| `target_char` | The specific target character (singleton or geminate) |
| `target_char_idx` | Character index of target in transcript string |
| `is_geminate` | 1 for geminate, 0 for singleton |

### Step 2 — Run

```bash
python run_experiment2.py \
    --model wav2vec2-base \
    --device cuda \
    --epochs 8 \
    --train-subset 1000 \
    --diag-csv path/to/diagnostic_metadata.csv
```

To skip training and use existing checkpoints:
```bash
python run_experiment2.py \
    --model wav2vec2-base \
    --skip-training \
    --diag-csv path/to/diagnostic_metadata.csv
```

Results saved to:
- `logs/ctc/wav2vec2-base/layer_N.pt` — per-layer CTC head checkpoints
- `data/arabic-geminate/results/wav2vec2-base_ctc_duration_summary.csv` — mean duration per layer and category
- `logs/ctc/forced_alignment_results.csv` — raw per-token durations
