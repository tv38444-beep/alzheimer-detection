"""
Train every extractor / selector / classifier combination for the web app,
on YOUR Alzheimer's brain-scan dataset, and write the files into the model
folder configured in app.py (models_retrained/extractors, selectors, classifiers).

Run it on a GPU (Google Colab: Runtime > Change runtime type > GPU). It also
works on CPU, just slowly during fine-tuning.

    !pip install -q torch torchvision scikit-learn xgboost pandas joblib
    !python train_models.py --data /content/alzheimer_dataset

Useful switches (all optional, config block below has the same settings):
    --quick                     tiny/fast run to check everything works end to end
    --only efficientnet_b0      train just one (or more) extractors
    --no-reuse                  re-fine-tune the CNNs even if weights already exist
    --levels 10 20 30 40 50 60  percent levels offered by the web-page slider

HOW THE PIPELINE IS BUILT (and why accuracy holds when you use 10-60%)
----------------------------------------------------------------------
 scan -> CNN backbone (fine-tuned) -> raw features (e.g. 1280)
      -> standardise -> selector keeps the best ~600-900 features, RANKED best-first
      -> user picks 10-60%: the TOP k% of those ranked features go to the classifier
      -> classifier -> Alzheimer's / Non-Alzheimer's

The classifier for each percent level is trained on exactly those top-k features
(not on all features with some zeroed out). That is why a 10%-60% budget stays
accurate: the model has seen that exact input size during training. Test-set
metrics for every (extractor, selector, classifier, level) are saved to
OUT_DIR/metrics.json, and the web page shows them next to the prediction.
"""
import argparse
import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import models, transforms
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG - edit this block (or use the command-line switches above)
# ============================================================================
DATA_DIR = Path("/content/alzheimer_dataset")   # folder that contains one sub-folder per class
OUT_DIR = Path("models_retrained")               # app.py loads this completed model set
CACHE_DIR = Path("features_cache")               # extracted features are cached here

EXTRACTORS_TO_TRAIN = ["resnet50", "efficientnet_b0", "convnext_tiny", "densenet121"]
SELECTORS_TO_TRAIN = ["pca", "mrmr", "pso", "rfe"]
CLASSIFIERS_TO_TRAIN = ["svm", "random_forest", "knn", "xgboost"]

# Selector output size: SELECT_FRACTION of the extractor's feature count, capped at
# SELECT_MAX. 1280 -> 900, 2048 -> 900, 1024 -> 819, 768 -> 614 (all in 600-900).
SELECT_FRACTION = 0.80
SELECT_MAX = 900

# Percent of the selected features sent to the classifier (the web-page slider).
# One classifier is trained per level. Use range(10, 61, 5) for a finer slider.
PERCENT_LEVELS = [10, 20, 30, 40, 50, 60]
TUNE_LEVEL = 30           # hyper-parameters are tuned once here, then reused for every level

# --- CNN fine-tuning (transfer learning) ---
FINE_TUNE = True
REUSE_FINETUNED_WEIGHTS = True    # reuse OUT_DIR/extractors/<name>.pth if it exists (same dataset/split!)
UNFREEZE = "all"                  # "all" = best accuracy, "last_block" = faster
FINE_TUNE_EPOCHS = 1  # one CPU-feasible transfer-learning pass per backbone
BATCH_SIZE = 32
LR_BACKBONE, LR_HEAD = 3e-5, 1e-3
IMG_SIZE = 224

# --- speed controls for the selectors / classifiers (rows used, not features) ---
MI_ROWS, RFE_ROWS, PSO_ROWS = 4000, 4000, 3000
TUNE_ROWS = 4000
SVM_MAX_ROWS = 6000       # RBF-SVM with probabilities is O(n^2); more rows rarely help
RFE_STEP = 0.12           # RFE drops 12% of the remaining features per round
PSO_PARTICLES, PSO_ITERS = 16, 10
QUICK_MODE = False
USE_FEATURE_CACHE = True
RANDOM_STATE = 42

# Folder name -> label (0 = Non-Alzheimer's, 1 = Alzheimer's). Unknown folder names are
# matched by keyword ("non..." -> 0; "mild/moderate/demented/alzheimer" -> 1) and printed.
CLASS_MAP = {
    "NonDemented": 0, "non_demented": 0, "Non_Demented": 0, "Normal": 0, "CN": 0,
    "VeryMildDemented": 1, "very_mild_demented": 1, "Very_Mild_Demented": 1,
    "MildDemented": 1, "mild_demented": 1, "Mild_Demented": 1,
    "ModerateDemented": 1, "moderate_demented": 1, "Moderate_Demented": 1,
    # Four-class Combined Dataset used in this project; app.py exposes binary AD detection.
    "No Impairment": 0, "Mild Impairment": 1,
    "Moderate Impairment": 1, "Very Mild Impairment": 1,
    "AD": 1,
}

MODEL_FORMAT = 3          # must match app.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 1. Dataset: index, label, split WITHOUT leakage
# ============================================================================
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def label_for_folder(name):
    if name in CLASS_MAP:
        return CLASS_MAP[name]
    n = re.sub(r"[^a-z]", "", name.lower())
    if n in {"noimpairment", "nonimpairment"}:
        return 0
    if n in {"mildimpairment", "moderateimpairment", "verymildimpairment"}:
        return 1
    if n.startswith(("non", "normal", "control", "healthy")):
        return 0
    if n == "ad" or any(t in n for t in ("mild", "moderate", "severe", "demented", "dementia", "alzheim")):
        return 1
    return None


def group_id(path: Path) -> str:
    """
    Augmented datasets (aug_00400_mild_13.jpg) hold several rotated/flipped copies of the
    SAME original image. Copies of one original must stay on the same side of every split
    (train / val / test AND inside cross-validation), otherwise accuracy is inflated.
    Group = class folder + trailing numeric id of the file name.
    """
    stem = path.stem
    tail = stem.split("_")[-1]
    key = tail if tail.isdigit() else stem
    return f"{path.parent.name}_{key}"


def index_dataset(root: Path):
    rows, skipped = [], {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        label = label_for_folder(p.parent.name)
        if label is None:
            skipped[p.parent.name] = skipped.get(p.parent.name, 0) + 1
            continue
        rows.append({"path": str(p), "path_rel": p.relative_to(root).as_posix(),
                     "class_folder": p.parent.name, "label": label, "group": group_id(p)})
    if skipped:
        print("WARNING - folders skipped (no label rule):", skipped)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No images found under {root}. Check --data / DATA_DIR and CLASS_MAP.")
    return df


def load_and_split():
    split_csv = OUT_DIR / "split.csv"
    # Preserve an existing train/test directory split; validation comes only from train.
    presplit_train = DATA_DIR / "train"
    presplit_test = DATA_DIR / "test"
    if presplit_train.is_dir() and presplit_test.is_dir():
        print(f"Using the supplied train/test split under {DATA_DIR}")
        train_all = index_dataset(presplit_train)
        test_df = index_dataset(presplit_test)
        print("Train counts:\n", train_all.groupby(["class_folder", "label"]).size().to_string())
        print("Test counts:\n", test_df.groupby(["class_folder", "label"]).size().to_string())
        g = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
        tr_idx, va_idx = next(g.split(train_all, groups=train_all["group"]))
        train_df, val_df = train_all.iloc[tr_idx].copy(), train_all.iloc[va_idx].copy()
        for frame, split in ((train_df, "train"), (val_df, "val"), (test_df, "test")):
            frame["path_rel"] = frame["path"].map(lambda p: Path(p).relative_to(DATA_DIR).as_posix())
            frame["split"] = split
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.concat([train_df, val_df, test_df])[["path_rel", "label", "group", "split"]].to_csv(split_csv, index=False)
        print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)} (provided test set untouched)")
        return tuple(x.reset_index(drop=True) for x in (train_df, val_df, test_df))

    if REUSE_FINETUNED_WEIGHTS and split_csv.exists():
        sp = pd.read_csv(split_csv)
        sp["path"] = [str(DATA_DIR / p) for p in sp["path_rel"]]
        if not all(Path(p).is_file() for p in sp["path"]):
            raise SystemExit(f"{split_csv} points to files that are missing from DATA_DIR.")
        print(f"Re-using the saved train/val/test split from {split_csv}")
        return tuple(sp[sp["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))

    print("Indexing dataset ...")
    df = index_dataset(DATA_DIR)
    print(df.groupby(["class_folder", "label"]).size().rename("images").to_string())
    if QUICK_MODE:
        df = pd.concat([g.sample(min(len(g), 700), random_state=RANDOM_STATE) for _, g in df.groupby("label")])
        print("QUICK_MODE: using", len(df), "images")

    # deterministic: depends only on the set of group ids + RANDOM_STATE
    g1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    tr_idx, rest_idx = next(g1.split(df, groups=df["group"]))
    rest = df.iloc[rest_idx]
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    va_idx, te_idx = next(g2.split(rest, groups=rest["group"]))
    train_df, val_df, test_df = df.iloc[tr_idx], rest.iloc[va_idx], rest.iloc[te_idx]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")])[
        ["path_rel", "label", "group", "split"]].to_csv(split_csv, index=False)
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}  (split by original image, not by file)")
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True))


# ============================================================================
# 2. CNN backbones (transfer learning)
# ============================================================================
NORM_MEAN, NORM_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


class ScanDataset(torch.utils.data.Dataset):
    def __init__(self, frame, tf):
        self.frame, self.tf = frame.reset_index(drop=True), tf

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        return self.tf(Image.open(row["path"]).convert("RGB")), int(row["label"])


def loader(frame, tf, shuffle):
    return torch.utils.data.DataLoader(ScanDataset(frame, tf), batch_size=BATCH_SIZE, shuffle=shuffle,
                                       num_workers=2, pin_memory=(DEVICE.type == "cuda"))


_BUILDERS = {"resnet50": models.resnet50, "efficientnet_b0": models.efficientnet_b0,
             "convnext_tiny": models.convnext_tiny, "densenet121": models.densenet121}
_FEAT_DIM = {"resnet50": 2048, "efficientnet_b0": 1280, "convnext_tiny": 768, "densenet121": 1024}


def build_backbone(key, pretrained=True):
    m = _BUILDERS[key](weights="DEFAULT" if pretrained else None)
    if key == "resnet50":
        m.fc = nn.Identity()
    elif key == "efficientnet_b0":
        m.classifier = nn.Identity()
    elif key == "convnext_tiny":
        m.classifier[2] = nn.Identity()
    elif key == "densenet121":
        m.classifier = nn.Identity()
    return m, _FEAT_DIM[key]


class HeadedModel(nn.Module):
    """Backbone + small head, used only while fine-tuning."""

    def __init__(self, key):
        super().__init__()
        self.backbone, dim = build_backbone(key, pretrained=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 2))

    def forward(self, x):
        return self.head(self.backbone(x))


def set_trainable(key, model):
    for p in model.parameters():
        p.requires_grad = (UNFREEZE == "all")
    if UNFREEZE != "all":
        tails = {"resnet50": [model.backbone.layer4],
                 "efficientnet_b0": [model.backbone.features[-1]],
                 "convnext_tiny": [model.backbone.features[-1]],
                 "densenet121": [model.backbone.features.denseblock4, model.backbone.features.norm5]}
        for block in tails[key]:
            for p in block.parameters():
                p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True


@torch.no_grad()
def predict_labels(model, frame):
    model.eval()
    preds, ys = [], []
    for xb, yb in loader(frame, eval_tf, False):
        preds.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def fine_tune(key, train_df, val_df, test_df):
    print(f"\n=== Fine-tuning {key} (unfreeze={UNFREEZE}, epochs={FINE_TUNE_EPOCHS}) ===")
    model = HeadedModel(key).to(DEVICE)
    set_trainable(key, model)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": LR_BACKBONE},
        {"params": model.head.parameters(), "lr": LR_HEAD}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FINE_TUNE_EPOCHS)
    counts = np.bincount(train_df["label"].values, minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    tl = loader(train_df, train_tf, True)
    best_val, best_state = -1.0, None
    for epoch in range(FINE_TUNE_EPOCHS):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        yv, pv = predict_labels(model, val_df)
        val_acc = balanced_accuracy_score(yv, pv)
        print(f"  epoch {epoch + 1}/{FINE_TUNE_EPOCHS}  val balanced-acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val, best_state = val_acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    yt, pt = predict_labels(model, test_df)
    deep = {"accuracy": round(float(accuracy_score(yt, pt)), 4),
            "f1": round(float(f1_score(yt, pt, zero_division=0)), 4),
            "recall": round(float(recall_score(yt, pt, zero_division=0)), 4),
            "val_balanced_accuracy": round(float(best_val), 4)}
    print(f"  deep model (CNN head) TEST accuracy={deep['accuracy']:.4f}  f1={deep['f1']:.4f}")
    torch.save(model.backbone.state_dict(), OUT_DIR / "extractors" / f"{key}.pth")
    return model.backbone.eval(), deep


def get_backbone(key, train_df, val_df, test_df):
    weights = OUT_DIR / "extractors" / f"{key}.pth"
    if REUSE_FINETUNED_WEIGHTS and weights.exists():
        print(f"\n=== {key}: re-using fine-tuned weights {weights} ===")
        bb, _ = build_backbone(key, pretrained=False)
        bb.load_state_dict(torch.load(weights, map_location="cpu"))
        return bb.eval(), None, False
    if FINE_TUNE:
        bb, deep = fine_tune(key, train_df, val_df, test_df)
        return bb, deep, True
    bb, _ = build_backbone(key, pretrained=True)
    torch.save(bb.state_dict(), weights)
    return bb.eval(), None, True


@torch.no_grad()
def extract_features(backbone, frame):
    backbone.eval().to(DEVICE)
    feats, labels = [], []
    for xb, yb in loader(frame, eval_tf, False):
        feats.append(backbone(xb.to(DEVICE)).float().cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(feats).astype(np.float32), np.concatenate(labels)


# ============================================================================
# 3. Feature selectors. Every selector returns the best N features RANKED
#    best-first ("order"), so "top k%" is simply the first k columns.
# ============================================================================
def n_selected_for(dim):
    return int(max(10, min(SELECT_MAX, round(SELECT_FRACTION * dim), dim)))


def level_k(level, n):
    return int(max(1, round(level / 100.0 * n)))


def rank01(a):
    r = np.argsort(np.argsort(a)).astype(float)
    return r / max(len(a) - 1, 1)


def stratified_rows(y, n_rows, rng):
    if len(y) <= n_rows:
        return np.arange(len(y))
    take = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        take.append(rng.choice(idx, max(1, int(round(n_rows * len(idx) / len(y)))), replace=False))
    return np.sort(np.concatenate(take))


def mi_scores(Xs, y):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = stratified_rows(y, MI_ROWS, rng)
    return mutual_info_classif(Xs[rows], y[rows], n_neighbors=3, random_state=RANDOM_STATE)


def fit_pca(Xs, y, N, rel):
    n_comp = int(min(N, Xs.shape[0] - 1, Xs.shape[1]))
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(Xs)
    return {"kind": "pca", "pca": pca, "order": None, "importance": pca.explained_variance_ratio_}


def fit_mrmr(Xs, y, N, rel):
    """Greedy mRMR: relevance = mutual information with the label,
    redundancy = mean |correlation| with the features already chosen."""
    d = Xs.shape[1]
    rows = stratified_rows(y, MI_ROWS, np.random.default_rng(RANDOM_STATE))
    corr = np.abs(np.nan_to_num(np.corrcoef(Xs[rows].T)))
    np.fill_diagonal(corr, 0.0)
    rel01 = rel / (rel.max() + 1e-12)
    avail, red, chosen, at = np.ones(d, bool), np.zeros(d), [], []
    for step in range(N):
        score = rel01 - (red / step if step else 0.0)
        score = np.where(avail, score, -np.inf)
        j = int(score.argmax())
        chosen.append(j)
        at.append(score[j])
        avail[j] = False
        red += corr[:, j]
    imp = np.asarray(at) - np.min(at) + 1e-3
    return {"kind": "rank", "pca": None, "order": np.array(chosen, dtype=np.int64),
            "importance": np.minimum.accumulate(imp)}


def fit_rfe(Xs, y, N, rel):
    """Recursive feature elimination with a logistic-regression ranker. The full
    elimination order is recorded, so the survivors are ranked best-first."""
    d = Xs.shape[1]
    rows = stratified_rows(y, RFE_ROWS, np.random.default_rng(RANDOM_STATE))
    Xr, yr = Xs[rows], y[rows]
    remaining, dropped, drop_imp = np.arange(d), [], {}
    while remaining.size > 1:
        clf = LogisticRegression(C=0.05, max_iter=150, class_weight="balanced").fit(Xr[:, remaining], yr)
        w = np.abs(clf.coef_[0])
        wn = w / (w.max() + 1e-12)
        n_drop = min(max(1, int(remaining.size * RFE_STEP)), remaining.size - 1)
        worst = np.argsort(w)[:n_drop]
        for pos in worst:
            drop_imp[int(remaining[pos])] = float(wn[pos])
        dropped.extend(remaining[worst].tolist())
        remaining = np.delete(remaining, worst)
    dropped.extend(remaining.tolist())
    drop_imp[int(remaining[0])] = 1.0
    order = np.array(dropped[::-1], dtype=np.int64)[:N]
    imp = np.array([drop_imp[int(i)] for i in order]) + 1e-3
    return {"kind": "rank", "pca": None, "order": order, "importance": np.minimum.accumulate(imp)}


def fit_pso(Xs, y, N, rel):
    """Binary particle-swarm wrapper search. The swarm looks for a subset of about N
    features that a quick linear model classifies well; features inside the final
    subset are then ranked by swarm votes + mutual information."""
    d = Xs.shape[1]
    rng = np.random.default_rng(RANDOM_STATE)
    rows = stratified_rows(y, PSO_ROWS, rng)
    perm = rng.permutation(len(rows))
    cut = int(0.7 * len(rows))
    tr, va = rows[perm[:cut]], rows[perm[cut:]]
    Xtr_, ytr_, Xva_, yva_ = Xs[tr], y[tr], Xs[va], y[va]

    def fitness(mask):
        n = int(mask.sum())
        if n < 5:
            return -1.0
        clf = LogisticRegression(C=0.1, max_iter=60, class_weight="balanced").fit(Xtr_[:, mask], ytr_)
        return balanced_accuracy_score(yva_, clf.predict(Xva_[:, mask])) - 0.5 * abs(n - N) / d

    rel01 = rel / (rel.max() + 1e-12)
    p_incl = np.clip((N / d) * (0.5 + rel01), 0.05, 0.98)
    P = PSO_PARTICLES
    pos = np.zeros((P, d))
    pos[: P // 2] = rng.random((P // 2, d)) < (N / d)
    pos[P // 2:] = rng.random((P - P // 2, d)) < p_incl
    vel = rng.normal(0, 0.5, (P, d))
    pbest = pos.copy()
    pbest_score = np.array([fitness(p > 0.5) for p in pos])
    g = int(pbest_score.argmax())
    gbest, gbest_score = pbest[g].copy(), pbest_score[g]
    votes = np.zeros(d)
    for _ in range(PSO_ITERS):
        r1, r2 = rng.random((P, d)), rng.random((P, d))
        vel = np.clip(0.7 * vel + 1.5 * r1 * (pbest - pos) + 1.5 * r2 * (gbest - pos), -4, 4)
        pos = (rng.random((P, d)) < 1.0 / (1.0 + np.exp(-vel))).astype(float)
        scores = np.array([fitness(p > 0.5) for p in pos])
        better = scores > pbest_score
        pbest[better], pbest_score[better] = pos[better], scores[better]
        if pbest_score.max() > gbest_score:
            g = int(pbest_score.argmax())
            gbest, gbest_score = pbest[g].copy(), pbest_score[g]
        votes += pbest.mean(axis=0)
    combined = rank01(votes) + rank01(rel)
    chosen = np.flatnonzero(gbest > 0.5)
    if chosen.size > N:                      # trim to exactly N, keeping the best-scored
        chosen = chosen[np.argsort(-combined[chosen])[:N]]
    elif chosen.size < N:                    # pad with the best-scored unselected features
        rest = np.setdiff1d(np.arange(d), chosen)
        chosen = np.concatenate([chosen, rest[np.argsort(-combined[rest])[: N - chosen.size]]])
    order = chosen[np.argsort(-combined[chosen])].astype(np.int64)
    imp = combined[order] - combined[order].min() + 1e-3
    return {"kind": "rank", "pca": None, "order": order, "importance": imp}


SELECTOR_FITTERS = {"pca": fit_pca, "mrmr": fit_mrmr, "pso": fit_pso, "rfe": fit_rfe}


def apply_bundle(bundle, X):
    """Raw CNN features -> N ranked features. Identical to apply_selector() in app.py."""
    Xs = bundle["scaler"].transform(X)
    Z = bundle["pca"].transform(Xs) if bundle["kind"] == "pca" else Xs[:, bundle["order"]]
    return Z.astype(np.float32)


# ============================================================================
# 4. Classifiers (one per percent level, trained on exactly the top-k features)
# ============================================================================
def make_estimator(name, pos_weight):
    if name == "svm":
        return SVC(probability=True, class_weight="balanced", cache_size=1000, random_state=RANDOM_STATE)
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)
    if name == "knn":
        return KNeighborsClassifier(n_jobs=-1)
    return XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=-1,
                         scale_pos_weight=pos_weight, random_state=RANDOM_STATE)


def param_grid(name):
    if QUICK_MODE:
        return {"svm": {"C": [1, 10]}, "random_forest": {"n_estimators": [200]},
                "knn": {"n_neighbors": [5, 9]}, "xgboost": {"n_estimators": [200]}}[name]
    return {"svm": {"C": [1, 10], "gamma": ["scale"]},
            "random_forest": {"n_estimators": [300], "max_depth": [None, 20]},
            "knn": {"n_neighbors": [3, 5, 9, 15], "weights": ["uniform", "distance"]},
            "xgboost": {"n_estimators": [300, 500], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}}[name]


def tune(name, Z, y, groups, pos_weight):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = stratified_rows(y, TUNE_ROWS, rng)
    cv = GroupKFold(n_splits=3)       # group-aware: augmented copies never straddle folds
    search = GridSearchCV(make_estimator(name, pos_weight), param_grid(name), cv=cv,
                          scoring="balanced_accuracy", n_jobs=(-1 if name in ("svm", "knn") else 1))
    search.fit(Z[rows], y[rows], groups=groups[rows])
    return search.best_params_


def evaluate(model, Zte, yte):
    pred = model.predict(Zte)
    out = {"accuracy": accuracy_score(yte, pred), "balanced_accuracy": balanced_accuracy_score(yte, pred),
           "precision": precision_score(yte, pred, zero_division=0),
           "recall": recall_score(yte, pred, zero_division=0), "f1": f1_score(yte, pred, zero_division=0)}
    try:
        out["auc"] = roc_auc_score(yte, model.predict_proba(Zte)[:, 1])
    except Exception:  # noqa: BLE001
        out["auc"] = None
    out = {k: (None if v is None else round(float(v), 4)) for k, v in out.items()}
    out["confusion_matrix"] = confusion_matrix(yte, pred, labels=[0, 1]).tolist()
    out["n_test"] = int(len(yte))
    return out


# ============================================================================
# 5. Orchestration
# ============================================================================
def write_metrics(meta, results, deep):
    payload = {"generated": datetime.now().isoformat(timespec="seconds"), "format": MODEL_FORMAT,
               "levels": PERCENT_LEVELS, **meta, "deep": deep, "results": results}
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")


def train_downstream(ext, Xtr, ytr, gtr, Xte, yte, results, selected_dims):
    """Everything after feature extraction: scaler -> selectors -> classifiers x levels."""
    dim = Xtr.shape[1]
    N = n_selected_for(dim)
    scaler = StandardScaler().fit(Xtr)
    Xs = scaler.transform(Xtr).astype(np.float32)
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"[{ext}] {dim} raw features -> selectors keep up to {N}")
    rel = mi_scores(Xs, ytr)
    rows_out = []
    for sel in SELECTORS_TO_TRAIN:
        t0 = time.time()
        fit = SELECTOR_FITTERS[sel](Xs, ytr, N, rel)
        bundle = {"version": MODEL_FORMAT, "kind": fit["kind"], "scaler": scaler, "pca": fit["pca"],
                  "order": fit["order"], "importance": np.asarray(fit["importance"], dtype=float),
                  "dim_in": dim, "n_selected": int(len(fit["importance"])), "extractor": ext, "selector": sel}
        joblib.dump(bundle, OUT_DIR / "selectors" / f"{ext}__{sel}.pkl", compress=3)
        n_eff = bundle["n_selected"]
        selected_dims[f"{ext}|{sel}"] = n_eff
        Ztr, Zte = apply_bundle(bundle, Xtr), apply_bundle(bundle, Xte)
        print(f"  [{ext}] selector {sel:5s}: {dim} -> {n_eff} features ({time.time() - t0:.0f}s)")
        for clf in CLASSIFIERS_TO_TRAIN:
            k_tune = level_k(TUNE_LEVEL, n_eff)
            best = tune(clf, Ztr[:, :k_tune], ytr, gtr, pos_weight)
            for level in PERCENT_LEVELS:
                k = level_k(level, n_eff)
                est = make_estimator(clf, pos_weight).set_params(**best)
                rows = (stratified_rows(ytr, SVM_MAX_ROWS, np.random.default_rng(RANDOM_STATE))
                        if clf == "svm" else np.arange(len(ytr)))
                est.fit(Ztr[rows][:, :k], ytr[rows])
                joblib.dump({"version": MODEL_FORMAT, "model": est, "k": k, "level": level, "n_selected": n_eff,
                             "extractor": ext, "selector": sel, "classifier": clf},
                            OUT_DIR / "classifiers" / f"{ext}__{sel}__{clf}__p{level}.pkl", compress=3)
                m = evaluate(est, Zte[:, :k], yte)
                m["k"] = k
                results[f"{ext}|{sel}|{clf}|{level}"] = m
                rows_out.append({"extractor": ext, "selector": sel, "classifier": clf, "percent": level,
                                 "features_used": k, "features_selected": n_eff,
                                 **{x: m[x] for x in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc")},
                                 "best_params": json.dumps(best)})
            at = results[f"{ext}|{sel}|{clf}|{PERCENT_LEVELS[-1]}"]
            lo = results[f"{ext}|{sel}|{clf}|{PERCENT_LEVELS[0]}"]
            print(f"    {clf:13s} acc@{PERCENT_LEVELS[0]}%={lo['accuracy']:.3f}  "
                  f"acc@{PERCENT_LEVELS[-1]}%={at['accuracy']:.3f}  f1@{PERCENT_LEVELS[-1]}%={at['f1']:.3f}  {best}")
    return rows_out


def features_for(ext, backbone, retrain, train_df, val_df, test_df):
    cache = CACHE_DIR / f"{ext}.npz"
    if USE_FEATURE_CACHE and not retrain and cache.exists():
        # Earlier cache files stored group IDs as object arrays. Those IDs are
        # generated locally by this script, so allow NumPy to read that field.
        z = np.load(cache, allow_pickle=True)
        print(f"[{ext}] loaded cached features from {cache}")
        return z["Xtr"], z["ytr"], z["gtr"], z["Xte"], z["yte"]
    trval = pd.concat([train_df, val_df], ignore_index=True)
    print(f"[{ext}] extracting features ...")
    Xtr, ytr = extract_features(backbone, trval)
    Xte, yte = extract_features(backbone, test_df)
    gtr = trval["group"].astype(str).values
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, gtr=np.asarray(gtr, dtype=str), Xte=Xte, yte=yte)
    return Xtr, ytr, gtr, Xte, yte


def apply_cli():
    global DATA_DIR, OUT_DIR, CACHE_DIR, QUICK_MODE, REUSE_FINETUNED_WEIGHTS, EXTRACTORS_TO_TRAIN, PERCENT_LEVELS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--output", type=Path, help="write a complete new model set here (default: models_retrained)")
    ap.add_argument("--cache", type=Path, help="feature-cache directory (default: features_cache)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-reuse", action="store_true")
    ap.add_argument("--only", nargs="+", choices=list(_BUILDERS))
    ap.add_argument("--levels", nargs="+", type=int)
    a = ap.parse_args()
    if a.data:
        DATA_DIR = a.data
    if a.output:
        OUT_DIR = a.output
    if a.cache:
        CACHE_DIR = a.cache
    if a.quick:
        QUICK_MODE = True
    if a.no_reuse:
        REUSE_FINETUNED_WEIGHTS = False
    if a.only:
        EXTRACTORS_TO_TRAIN = a.only
    if a.levels:
        PERCENT_LEVELS = sorted(set(a.levels))


def main():
    apply_cli()
    print("Device:", DEVICE)
    for d in ("extractors", "selectors", "classifiers"):
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = load_and_split()
    meta = {"dataset": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
            "select_max": SELECT_MAX, "select_fraction": SELECT_FRACTION}
    # Keep metrics from earlier --only runs when they describe the same dataset
    # and model configuration. This makes an interrupted multi-extractor run resumable.
    previous = {}
    metrics_path = OUT_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
    compatible = (previous.get("format") == MODEL_FORMAT
                  and previous.get("dataset") == meta["dataset"]
                  and previous.get("select_max") == SELECT_MAX
                  and previous.get("select_fraction") == SELECT_FRACTION
                  and previous.get("levels") == PERCENT_LEVELS)
    results = dict(previous.get("results", {})) if compatible else {}
    deep = dict(previous.get("deep", {})) if compatible else {}
    selected_dims = dict(previous.get("selected_dims", {})) if compatible else {}
    table = []
    for ext in EXTRACTORS_TO_TRAIN:
        t0 = time.time()
        backbone, deep_m, retrain = get_backbone(ext, train_df, val_df, test_df)
        if deep_m:
            deep[ext] = deep_m
        Xtr, ytr, gtr, Xte, yte = features_for(ext, backbone, retrain, train_df, val_df, test_df)
        table += train_downstream(ext, Xtr, ytr, gtr, Xte, yte, results, selected_dims)
        for row in table:
            key = f"{row['extractor']}|{row['selector']}|{row['classifier']}|{row['percent']}"
            if key in results:
                results[key]["best_params"] = row["best_params"]
        write_metrics({**meta, "selected_dims": selected_dims}, results, deep)   # usable after every extractor
        csv_rows = []
        for key, metric in results.items():
            extractor, selector, classifier, percent = key.split("|")
            csv_rows.append({"extractor": extractor, "selector": selector, "classifier": classifier,
                             "percent": int(percent), "features_used": metric.get("k"),
                             "features_selected": selected_dims.get(f"{extractor}|{selector}"),
                             **{name: metric.get(name) for name in
                                ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc")},
                             "best_params": metric.get("best_params", "")})
        pd.DataFrame(csv_rows).sort_values("f1", ascending=False).to_csv("results.csv", index=False)
        print(f"[{ext}] finished in {(time.time() - t0) / 60:.1f} min\n")

    res = pd.DataFrame(table)
    print("================ TOP 10 BY F1 (all levels) ================")
    print(res.sort_values("f1", ascending=False).head(10)[
        ["extractor", "selector", "classifier", "percent", "accuracy", "recall", "f1"]].to_string(index=False))
    print("\n================ BEST PIPELINE AT EACH PERCENT LEVEL ================")
    for level in PERCENT_LEVELS:
        b = res[res["percent"] == level].sort_values("f1", ascending=False).iloc[0]
        print(f"  {level:>3d}%  {b.extractor} -> {b.selector} -> {b.classifier}  "
              f"accuracy={b.accuracy:.3f}  recall={b.recall:.3f}  f1={b.f1:.3f}")
    print(f"\nDone. Copy the whole '{OUT_DIR}' folder into your VS Code project and run app.py.")


if __name__ == "__main__":
    main()
