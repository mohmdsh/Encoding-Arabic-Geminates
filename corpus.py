import re
import math
import random
from pathlib import Path
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from config import (
    WAV_DIR, TEST_WAV_DIR, TG_DIR, TEST_TG_DIR,
    DATA_DIR, SILENCE_LABELS, TARGET_PHONES, GEMINATE_PHONES,
    CLASS_GROUPS, TRAIN_RATIO, SAMPLE_RATE,
)

BW_TO_ARABIC = {
    "A":"\u0627","b":"\u0628","t":"\u062a","^":"\u062b","j":"\u062c","H":"\u062d",
    "x":"\u062e","d":"\u062f","*":"\u0630","r":"\u0631","z":"\u0632","s":"\u0633",
    "$":"\u0634","S":"\u0635","D":"\u0636","T":"\u0637","Z":"\u0638","E":"\u0639",
    "g":"\u063a","f":"\u641","q":"\u0642","k":"\u0643","l":"\u0644","m":"\u0645",
    "n":"\u0646","h":"\u0647","w":"\u0648","y":"\u064a","'":"\u0621",
    ">":"\u0623","<":"\u0625","&":"\u0624","}":"\u0626","|":"\u0622",
    "{":"\u0671","`":"\u0670","Y":"\u0649","p":"\u0629",
    "a":"\u064e","u":"\u064f","i":"\u0650","F":"\u064b","N":"\u064c","K":"\u064d",
    "~":"\u0651","o":"\u0652","_":"\u0640",
}

HALABI_TO_STANDARD_BW = {"I":"<","O":">","W":"&"}

PHONE_TO_CLASS = {
    ph: (cls, sing)
    for cls, pairs in CLASS_GROUPS.items()
    for sing, gem in pairs
    for ph in (sing, gem)
}


def parse_textgrid_long(tg_path: Path) -> dict:
    text  = tg_path.read_text(encoding="utf-8", errors="replace")
    tiers = {}
    for block in re.split(r"item\s*\[\d+\]", text)[1:]:
        name_m = re.search(r'name\s*=\s*"(.*?)"', block)
        if not name_m:
            continue
        tier_name = name_m.group(1).strip()
        if not tier_name:
            continue
        intervals = re.findall(
            r"xmin\s*=\s*([\d.eE+\-]+).*?xmax\s*=\s*([\d.eE+\-]+).*?text\s*=\s*\"(.*?)\"",
            block, re.DOTALL,
        )
        rows = [{"start": float(s), "end": float(e), "label": t.strip()}
                for s, e, t in intervals]
        tiers[tier_name] = pd.DataFrame(rows)
    return tiers


def get_phone_df(tg_path: Path) -> pd.DataFrame:
    tiers = parse_textgrid_long(tg_path)
    df = tiers.get("phones", pd.DataFrame())
    if df.empty:
        return df
    df = df.rename(columns={"label": "phone"}).copy()
    df["duration_ms"] = (df["end"] - df["start"]) * 1000
    return df.reset_index(drop=True)


def get_word_df(tg_path: Path) -> pd.DataFrame:
    tiers = parse_textgrid_long(tg_path)
    df = tiers.get("words", pd.DataFrame())
    if df.empty:
        return df
    return df.rename(columns={"label": "word_bw"}).copy().reset_index(drop=True)


def halabi_word_to_arabic(word_bw: str) -> str:
    if not word_bw or word_bw in SILENCE_LABELS:
        return "[silence]"
    for h, s in HALABI_TO_STANDARD_BW.items():
        word_bw = word_bw.replace(h, s)
    return "".join(BW_TO_ARABIC.get(ch, ch) for ch in word_bw)


def get_consonants(phone_df: pd.DataFrame, file_id: str,
                   word_df: pd.DataFrame = None) -> pd.DataFrame:
    records = []
    for _, row in phone_df.iterrows():
        ph = row["phone"]
        if ph not in TARGET_PHONES:
            continue
        cls_name, sing_label = PHONE_TO_CLASS[ph]
        word_bw = ""
        if word_df is not None and not word_df.empty:
            span = word_df[
                (word_df["start"] <= row["start"] + 1e-4) &
                (word_df["end"]   >= row["end"]   - 1e-4)
            ]
            if not span.empty:
                word_bw = span.iloc[0]["word_bw"]
        records.append({
            "file_id":         file_id,
            "word_bw":         word_bw,
            "phone":           ph,
            "phone_class":     cls_name,
            "singleton_label": sing_label,
            "start":           row["start"],
            "end":             row["end"],
            "duration_ms":     row["duration_ms"],
            "is_geminate":     1 if ph in GEMINATE_PHONES else 0,
        })
    return pd.DataFrame(records)


def load_consonants_from_dir(tg_dir: Path, file_id_filter: set = None,
                              label: str = "") -> pd.DataFrame:
    tg_files = sorted(tg_dir.glob("*.TextGrid"))
    if file_id_filter is not None:
        tg_files = [f for f in tg_files if f.stem in file_id_filter]
    if not tg_files:
        raise FileNotFoundError(f"No matching TextGrid files in {tg_dir}")
    dfs = []
    for tg_path in tqdm(tg_files, desc=f"TextGrids ({label})"):
        file_id = tg_path.stem
        try:
            p_df = get_phone_df(tg_path)
            w_df = get_word_df(tg_path)
            cons = get_consonants(p_df, file_id, w_df)
            if not cons.empty:
                dfs.append(cons)
        except Exception as e:
            print(f"  SKIP {tg_path.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_train_val_split(wav_dir: Path, tg_dir: Path,
                           train_ratio: float = TRAIN_RATIO,
                           seed: int = 42) -> tuple:
    wav_files = sorted(wav_dir.glob("*.wav"))
    tg_files  = {f.stem for f in tg_dir.glob("*.TextGrid")}
    all_files = [f for f in wav_files if f.stem in tg_files]

    has_gem = set()
    for f in tqdm(all_files, desc="Checking geminates"):
        try:
            p_df = get_phone_df(tg_dir / f"{f.stem}.TextGrid")
            if any(ph in GEMINATE_PHONES for ph in p_df["phone"]):
                has_gem.add(f.stem)
        except Exception:
            pass

    gem_files    = [f for f in all_files if f.stem in has_gem]
    no_gem_files = [f for f in all_files if f.stem not in has_gem]
    rng = random.Random(seed)
    rng.shuffle(gem_files); rng.shuffle(no_gem_files)

    def split(lst, ratio):
        n = int(len(lst) * ratio)
        return lst[:n], lst[n:]

    train_gem, val_gem       = split(gem_files, train_ratio)
    train_no,  val_no        = split(no_gem_files, train_ratio)
    train_files = train_gem + train_no
    val_files   = val_gem   + val_no
    train_ids   = {f.stem for f in train_files}
    val_ids     = {f.stem for f in val_files}
    return train_files, val_files, train_ids, val_ids
