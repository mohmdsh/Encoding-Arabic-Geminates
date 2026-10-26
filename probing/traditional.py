import numpy as np
import pandas as pd
from pathlib import Path
from ast import literal_eval
from sklearn.linear_model  import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import f1_score, r2_score
from config import WAV_DIR, TEST_WAV_DIR, CLASS_GROUPS, RESULTS_DIR
from features.acoustic import TRAD_FEATS


def _get_matrix(df: pd.DataFrame, feat_fn) -> tuple:
    Xs, yc, yd = [], [], []
    for _, row in df.iterrows():
        wav = WAV_DIR / f"{row['file_id']}.wav"
        if not wav.exists():
            wav = TEST_WAV_DIR / f"{row['file_id']}.wav"
        if not wav.exists():
            continue
        frames = row.get("frame_indices", [])
        if isinstance(frames, str):
            frames = literal_eval(frames)
        fr, to = (int(frames[0]), int(frames[-1]) + 1) if frames else (0, 1)
        try:
            vec = feat_fn(str(wav), fr, to)
            Xs.append(vec)
            yc.append(1 if row["is_geminate"] else 0)
            yd.append(float(row["duration_ms"]))
        except Exception:
            continue
    if not Xs:
        return None, None, None
    return np.vstack(Xs), np.array(yc), np.array(yd)


def run_traditional_probes(train_df: pd.DataFrame,
                             val_df: pd.DataFrame,
                             save: bool = True) -> pd.DataFrame:
    rows = []
    for cls_name, pairs in CLASS_GROUPS.items():
        for sing, gem in pairs:
            tr = train_df[train_df["phone"].isin([sing, gem])]
            va = val_df[val_df["phone"].isin([sing, gem])]
            if len(tr) < 10 or len(va) < 4:
                continue
            for feat_name, feat_fn in TRAD_FEATS.items():
                X_tr, yc_tr, yd_tr = _get_matrix(tr, feat_fn)
                X_va, yc_va, yd_va = _get_matrix(va, feat_fn)
                if X_tr is None or X_va is None:
                    continue
                clf = Pipeline([
                    ("sc", StandardScaler()),
                    ("lr", LogisticRegression(
                        penalty="l1", solver="liblinear", C=1,
                        max_iter=5000, class_weight="balanced")),
                ]).fit(X_tr, yc_tr)
                yc_pred = clf.predict(X_va)
                f1g = f1_score(yc_va, yc_pred, pos_label=1, zero_division=0)
                f1s = f1_score(yc_va, yc_pred, pos_label=0, zero_division=0)

                reg = LinearRegression(n_jobs=-1)
                tr_gem = tr[tr["is_geminate"] == 1]
                va_gem = va[va["is_geminate"] == 1]
                Xg_tr, _, yd_g_tr = _get_matrix(tr_gem, feat_fn)
                Xg_va, _, yd_g_va = _get_matrix(va_gem, feat_fn)
                r2g = float("nan")
                if Xg_tr is not None and Xg_va is not None:
                    reg.fit(Xg_tr, yd_g_tr)
                    r2g = r2_score(yd_g_va, reg.predict(Xg_va))

                rows.append({
                    "class": cls_name, "pair": f"{sing}/{gem}",
                    "feature": feat_name,
                    "f1_gem": round(f1g, 4), "f1_sing": round(f1s, 4),
                    "r2_gem": round(r2g, 4),
                })

    results_df = pd.DataFrame(rows)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "traditional_probe_results.csv"
        results_df.to_csv(out, index=False)
        print(f"Saved -> {out}")
    return results_df
