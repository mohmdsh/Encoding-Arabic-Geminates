import math
import numpy as np
import pandas as pd
from pathlib import Path
from ast import literal_eval
from sklearn.linear_model   import LinearRegression, LogisticRegression
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline
from sklearn.dummy          import DummyClassifier, DummyRegressor
from sklearn.metrics        import f1_score, r2_score, balanced_accuracy_score
from config import N_BOOTSTRAP


def _bootstrap_r2(data_df: pd.DataFrame, n_iter: int = N_BOOTSTRAP):
    n, scores = len(data_df), []
    for i in range(n_iter):
        s = data_df.sample(n=n, replace=True, random_state=i)
        scores.append(r2_score(s["y_true"], s["y_pred"]))
    return scores, n


def _bootstrap_f1(data_df: pd.DataFrame, pos_label,
                   n_iter: int = N_BOOTSTRAP):
    n, scores = len(data_df), []
    for i in range(n_iter):
        s = data_df.sample(n=n, replace=True, random_state=i)
        scores.append(f1_score(s["y_true"], s["y_pred"],
                               pos_label=pos_label, zero_division=0))
    return scores, n


def _ci(true_score: float, boot_scores: list, n: int,
         alpha: float = 0.95) -> tuple:
    factor = alpha * (np.std(boot_scores) / math.sqrt(n))
    return (true_score - factor, true_score + factor)


def safe_load_npy(path: Path):
    try:
        arr = np.load(str(path), allow_pickle=False)
        return arr if arr.size > 0 else None
    except Exception:
        return None


def get_features(data: pd.DataFrame, feat_dir: Path,
                  regression: bool = True) -> tuple:
    feats, labels = [], []
    for file_id, group in data.groupby("file_id"):
        arr = safe_load_npy(feat_dir / f"{file_id}.npy")
        if arr is None:
            continue
        for _, row in group.iterrows():
            idx_list = row["frame_indices"]
            if isinstance(idx_list, str):
                idx_list = literal_eval(idx_list)
            idx_list = [i for i in idx_list if i < len(arr)]
            if not idx_list:
                continue
            feats.append(arr[idx_list, :].mean(axis=0).reshape(1, -1))
            labels.append(row["duration_ms"] if regression else row["is_geminate"])
    if not feats:
        raise ValueError(f"No features loaded from {feat_dir}")
    return np.concatenate(feats, axis=0), np.array(labels)


def duration_probe(X_tr, y_tr, X_va, y_va,
                    layer_i: int, log_dir: Path) -> dict:
    model = LinearRegression(n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_va)
    score  = r2_score(y_va, y_pred)
    res_df = pd.DataFrame({"y_true": y_va, "y_pred": y_pred})
    res_df.to_csv(log_dir / f"layer_{layer_i}.csv", index=False)
    ci     = _ci(score, *_bootstrap_r2(res_df))
    dummy  = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    rand   = r2_score(y_va, dummy.predict(X_va))
    return {"r2": score, "ci": ci, "rand_r2": rand}


def classification_probe(X_tr, y_tr, X_va, y_va,
                           layer_i: int, log_dir: Path) -> dict:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1", solver="liblinear", C=1,
            max_iter=5000, class_weight="balanced")),
    ])
    model.fit(X_tr, y_tr)
    y_pred  = model.predict(X_va)
    bal_acc = balanced_accuracy_score(y_va, y_pred)
    f1_gem  = f1_score(y_va, y_pred, pos_label=1, zero_division=0)
    f1_sing = f1_score(y_va, y_pred, pos_label=0, zero_division=0)
    res_df  = pd.DataFrame({"y_true": y_va, "y_pred": y_pred})
    res_df.to_csv(log_dir / f"layer_{layer_i}.csv", index=False)
    ci_gem  = _ci(f1_gem,  *_bootstrap_f1(res_df, 1))
    ci_sing = _ci(f1_sing, *_bootstrap_f1(res_df, 0))
    dummy   = DummyClassifier(strategy="stratified", random_state=42).fit(X_tr, y_tr)
    rp      = dummy.predict(X_va)
    return {
        "balanced_acc": bal_acc,
        "f1_gem":       f1_gem,  "ci_gem":  ci_gem,
        "f1_sing":      f1_sing, "ci_sing": ci_sing,
        "rand_f1_gem":  f1_score(y_va, rp, pos_label=1, zero_division=0),
        "rand_f1_sing": f1_score(y_va, rp, pos_label=0, zero_division=0),
    }


def run_pair_probes(sing: str, gem: str,
                    train_df: pd.DataFrame, val_df: pd.DataFrame,
                    feat_root: Path, log_base: Path,
                    num_layers: int, model_name: str) -> dict:
    pair_key = f"{sing}_{gem}"
    cls_log  = log_base / "classification" / pair_key / model_name / "linear"
    dg_log   = log_base / "duration_geminate"  / pair_key / model_name / "linear"
    ds_log   = log_base / "duration_singleton" / pair_key / model_name / "linear"
    for d in [cls_log, dg_log, ds_log]:
        d.mkdir(parents=True, exist_ok=True)

    tr_sing  = train_df[train_df["phone"] == sing]
    tr_gem   = train_df[train_df["phone"] == gem]
    va_sing  = val_df[val_df["phone"] == sing]
    va_gem   = val_df[val_df["phone"] == gem]
    tr_cls   = train_df[train_df["phone"].isin([sing, gem])]
    va_cls   = val_df[val_df["phone"].isin([sing, gem])]

    results = {}
    for layer_i in range(0, num_layers + 1):
        layer_dir = feat_root / f"layer-{layer_i}"
        if not layer_dir.exists():
            continue
        row = {}
        if len(tr_cls) >= 10 and len(va_cls) >= 4:
            try:
                X_tr, y_tr = get_features(tr_cls, layer_dir, regression=False)
                X_va, y_va = get_features(va_cls, layer_dir, regression=False)
                row["cls"] = classification_probe(X_tr, y_tr, X_va, y_va,
                                                   layer_i, cls_log)
            except ValueError:
                pass
        if len(tr_gem) >= 5 and len(va_gem) >= 2:
            try:
                X_tr, y_tr = get_features(tr_gem, layer_dir, regression=True)
                X_va, y_va = get_features(va_gem, layer_dir, regression=True)
                row["dur_gem"] = duration_probe(X_tr, y_tr, X_va, y_va,
                                                 layer_i, dg_log)
            except ValueError:
                pass
        if len(tr_sing) >= 10 and len(va_sing) >= 4:
            try:
                X_tr, y_tr = get_features(tr_sing, layer_dir, regression=True)
                X_va, y_va = get_features(va_sing, layer_dir, regression=True)
                row["dur_sing"] = duration_probe(X_tr, y_tr, X_va, y_va,
                                                  layer_i, ds_log)
            except ValueError:
                pass
        if row:
            results[layer_i] = row
    return results
