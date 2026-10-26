from pathlib import Path

REPO_ROOT    = Path(__file__).parent
CORPUS_DIR   = REPO_ROOT / "arabic-speech-corpus"
WAV_DIR      = CORPUS_DIR / "wav"
TEST_WAV_DIR = CORPUS_DIR / "test set" / "wav"
DATA_DIR     = REPO_ROOT / "data" / "arabic-geminate"
FEATS_DIR    = REPO_ROOT / "data" / "feats" / "arabic-geminate"
LOGS_DIR     = REPO_ROOT / "logs" / "arabic-geminate"
TG_DIR       = CORPUS_DIR / "textgrids"
TEST_TG_DIR  = CORPUS_DIR / "test set" / "textgrids"
RESULTS_DIR  = DATA_DIR / "results"

MODELS = {
    "wav2vec2-base": "facebook/wav2vec2-base",
    "hubert-base":   "facebook/hubert-base-ls960",
    "wavlm-base":    "microsoft/wavlm-base",
}

MODEL_NAME  = "wav2vec2-base"
SAMPLE_RATE = 16_000
TRAIN_RATIO = 0.85
N_BOOTSTRAP = 500

CLASS_GROUPS = {
    "Stops (plain)":    [("b","bb"),("t","tt"),("d","dd"),("k","kk"),("g","gg"),("q","qq")],
    "Stops (emphatic)": [("T","TT"),("D","DD"),("S","SS"),("Z","ZZ")],
    "Fricatives":       [("s","ss"),("f","ff"),("z","zz"),("x","xx"),("h","hh")],
    "Nasals":           [("n","nn"),("m","mm")],
    "Liquids & glides": [("l","ll"),("r","rr"),("w","ww"),("y","yy")],
}

SILENCE_LABELS  = {"sil", "sp", "SIL", ""}
TARGET_PHONES   = {ph for pairs in CLASS_GROUPS.values() for p in pairs for ph in p}
GEMINATE_PHONES = {gem for pairs in CLASS_GROUPS.values() for _, gem in pairs}

TRAD_SR         = 16_000
TRAD_N_FFT      = 400
TRAD_HOP_LENGTH = 160
