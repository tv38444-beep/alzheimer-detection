"""
Alzheimer's brain-scan classification server (Flask).

Pipeline - each stage is its own API call so the page can show every output:
    /api/upload    load the scan
    /api/extract   CNN feature extraction  (ResNet-50 | EfficientNet-B0 | ConvNeXt-tiny | DenseNet-121)
    /api/select    feature selection       (PCA | mRMR | PSO | RFE) -> ~600-900 features, ranked best-first
    /api/classify  classification          (SVM | Random Forest | KNN | XGBoost) on the TOP 10-60% of them

Run:  python app.py     then open http://127.0.0.1:5000
Models come from train_models.py (see README.md).
"""
import base64
import io
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

IMG_SIZE = 224                      # must match train_models.py
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

MODEL_FORMAT = 3                    # must match train_models.py
PERCENT_MIN, PERCENT_MAX = 10, 60   # slider range (percent of the selected features)

MAX_UPLOAD_MB = 15
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SESSION_TTL_S = 30 * 60
MAX_SESSIONS = 100

EXTRACTORS = {"resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0",
              "convnext_tiny": "ConvNeXt-tiny", "densenet121": "DenseNet-121"}
FEATURE_DIM = {"resnet50": 2048, "efficientnet_b0": 1280, "convnext_tiny": 768, "densenet121": 1024}
SELECTORS = {"pca": "PCA", "mrmr": "mRMR", "pso": "PSO", "rfe": "RFE"}
CLASSIFIERS = {"svm": "SVM", "random_forest": "Random Forest", "knn": "KNN", "xgboost": "XGBoost"}

# train_models.py writes 0 = Non-Alzheimer's, 1 = Alzheimer's
CLASS_INFO = {
    0: ("Non-Alzheimer's (normal)", False),
    1: ("Alzheimer's disease", True),
    "CN": ("Non-Alzheimer's (normal)", False), "NC": ("Non-Alzheimer's (normal)", False),
    "NORMAL": ("Non-Alzheimer's (normal)", False), "AD": ("Alzheimer's disease", True),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("alz")

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


class PipelineError(Exception):
    """An error whose message is safe and useful to show in the web page."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@app.errorhandler(PipelineError)
def _handle_pipeline_error(e):
    return jsonify(error=str(e)), e.status


@app.errorhandler(413)
def _too_large(_):
    return jsonify(error=f"Image is larger than {MAX_UPLOAD_MB} MB."), 413


# --------------------------------------------------------------------------- #
# Model files
# --------------------------------------------------------------------------- #
def _find(folder, stem):
    for ext in (".pkl", ".joblib", ".pth", ".pt"):
        p = MODELS_DIR / folder / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def extractor_weights(ext):
    return _find("extractors", ext)


def selector_path(ext, sel):
    return _find("selectors", f"{ext}__{sel}")


def classifier_path(ext, sel, clf, level):
    return _find("classifiers", f"{ext}__{sel}__{clf}__p{level}")


def scan_models():
    """Which selector / classifier files exist: ({ext: [sel]}, {"ext|sel": {clf: [levels]}})."""
    selectors = {e: [s for s in SELECTORS if selector_path(e, s)] for e in EXTRACTORS}
    classifiers = {}
    folder = MODELS_DIR / "classifiers"
    if folder.is_dir():
        for p in folder.iterdir():
            if p.suffix.lower() not in (".pkl", ".joblib"):
                continue
            parts = p.stem.split("__")
            if len(parts) != 4 or not parts[3].startswith("p") or not parts[3][1:].isdigit():
                continue
            e, s, c, lv = parts[0], parts[1], parts[2], int(parts[3][1:])
            if e in EXTRACTORS and s in SELECTORS and c in CLASSIFIERS:
                classifiers.setdefault(f"{e}|{s}", {}).setdefault(c, []).append(lv)
        for per_clf in classifiers.values():
            for levels in per_clf.values():
                levels.sort()
    return selectors, classifiers


_metrics_cache = {"mtime": None, "data": {}}


def load_metrics():
    p = MODELS_DIR / "metrics.json"
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {}
    if _metrics_cache["mtime"] != mtime:
        try:
            _metrics_cache["data"] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _metrics_cache["data"] = {}
        _metrics_cache["mtime"] = mtime
    return _metrics_cache["data"]


# --------------------------------------------------------------------------- #
# Stage 2: feature extractors (transfer-learning backbones, head removed)
# --------------------------------------------------------------------------- #
_BUILDERS = {"resnet50": models.resnet50, "efficientnet_b0": models.efficientnet_b0,
             "convnext_tiny": models.convnext_tiny, "densenet121": models.densenet121}


def _strip_head(key, m):
    if key == "resnet50":
        m.fc = nn.Identity()
    elif key == "efficientnet_b0":
        m.classifier = nn.Identity()
    elif key == "convnext_tiny":
        m.classifier[2] = nn.Identity()
    elif key == "densenet121":
        m.classifier = nn.Identity()


def _load_extractor(key):
    ctor = _BUILDERS[key]
    weights = extractor_weights(key)
    if weights is None:
        log.warning("No fine-tuned weights for %s; using plain ImageNet weights.", key)
        model = ctor(weights="DEFAULT")
        _strip_head(key, model)
        source = "ImageNet"
    else:
        model = ctor(weights=None)
        _strip_head(key, model)
        state = torch.load(weights, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, _unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise PipelineError(
                f"{weights.name} does not match the {EXTRACTORS[key]} architecture "
                f"({len(missing)} layers missing). Save it with torch.save(backbone.state_dict(), ...).", 500)
        source = "fine-tuned"
    model.eval().to(DEVICE)
    return model, source


_extractors = {}
_load_lock = threading.Lock()


def get_extractor(key):
    with _load_lock:
        if key not in _extractors:
            log.info("Loading extractor %s on %s", key, DEVICE)
            _extractors[key] = _load_extractor(key)
        return _extractors[key]


_transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                                 transforms.Normalize(NORM_MEAN, NORM_STD)])


def run_extractor(key, img):
    model, source = get_extractor(key)
    img_contrast = ImageOps.autocontrast(img.convert("L")).convert("RGB")
    x = _transform(img_contrast).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        feats = model(x)
    return feats.flatten(1).cpu().numpy().astype(np.float32), source


# --------------------------------------------------------------------------- #
# Stage 3 / 4: saved selector bundles and classifiers
# --------------------------------------------------------------------------- #
_pickles = {}


def load_pickle(path):
    key = (str(path), path.stat().st_mtime)
    with _load_lock:
        if key not in _pickles:
            log.info("Loading %s", path.name)
            _pickles[key] = joblib.load(path)
        return _pickles[key]


def _require_bundle(obj, path):
    if not (isinstance(obj, dict) and obj.get("version") == MODEL_FORMAT):
        raise PipelineError(
            f"{path.name} was made by an older version of train_models.py. "
            "Re-run train_models.py (it re-uses your fine-tuned CNN weights) and copy the new models/ folder.", 500)


def apply_selector(bundle, X):
    """Raw CNN features -> N ranked features (best first) + their importance. Same as train_models.apply_bundle."""
    scaler = bundle["scaler"]
    if X.shape[1] != scaler.n_features_in_:
        raise PipelineError(
            f"This selector expects {scaler.n_features_in_} features but the extractor produced {X.shape[1]}. "
            "The selector was trained on a different extractor.", 500)
    Xs = scaler.transform(X)
    Z = bundle["pca"].transform(Xs) if bundle["kind"] == "pca" else Xs[:, bundle["order"]]
    imp = np.asarray(bundle["importance"], dtype=float)
    return Z.astype(np.float32), imp


def class_info(c):
    if hasattr(c, "item"):
        c = c.item()
    info = CLASS_INFO.get(c)
    if info is None and isinstance(c, str):
        info = CLASS_INFO.get(c.strip().upper())
    return info or (str(c), False)


def predict(clf, X):
    classes = list(getattr(clf, "classes_", []))
    source = "predict_proba"
    if hasattr(clf, "predict_proba"):
        proba = np.asarray(clf.predict_proba(X))[0]
    elif hasattr(clf, "decision_function"):
        source = "decision_function"
        d = np.atleast_1d(np.asarray(clf.decision_function(X))[0])
        if d.size == 1:
            p1 = 1.0 / (1.0 + np.exp(-d[0]))
            proba = np.array([1.0 - p1, p1])
        else:
            e = np.exp(d - d.max())
            proba = e / e.sum()
    else:
        source = "predict"
        pred = clf.predict(X)[0]
        if not classes:
            raise PipelineError("Classifier has no predict_proba and no classes_.", 500)
        proba = np.array([1.0 if c == pred else 0.0 for c in classes])
    if not classes:
        classes = list(range(len(proba)))
    return classes, proba.astype(float), source


# --------------------------------------------------------------------------- #
# Session store (holds the image and intermediate features between stages)
# --------------------------------------------------------------------------- #
_sessions = {}
_sess_lock = threading.Lock()


def _prune(now):
    for t in [t for t, s in _sessions.items() if now - s["ts"] > SESSION_TTL_S]:
        del _sessions[t]
    while len(_sessions) > MAX_SESSIONS:
        del _sessions[min(_sessions, key=lambda t: _sessions[t]["ts"])]


def new_session(img):
    token = uuid.uuid4().hex
    now = time.time()
    with _sess_lock:
        _prune(now)
        _sessions[token] = {"ts": now, "img": img, "features": None, "selected": None,
                            "extractor": None, "selector": None}
    return token


def get_session(token):
    now = time.time()
    with _sess_lock:
        s = _sessions.get(token)
        if not s or now - s["ts"] > SESSION_TTL_S:
            raise PipelineError("Session expired. Load the scan again.", 410)
        s["ts"] = now
        return s


def _body(valid):
    data = request.get_json(silent=True) or {}
    method = data.get("method")
    if method not in valid:
        raise PipelineError("Unknown method.")
    return get_session(data.get("token")), method, data


def binned(values, bins=96, absolute=True):
    """Down-sample a vector to <= `bins` bars scaled to 0-1 (for the page's charts)."""
    v = np.asarray(values, dtype=float).ravel()
    v = np.abs(v) if absolute else v
    n = v.size
    bins = min(bins, n)
    edges = np.linspace(0, n, bins + 1).astype(int)
    bars = np.array([v[a:b].mean() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])
    if bars.max() > 0:
        bars = bars / bars.max()
    return [round(float(b), 4) for b in bars]


def summarize(vec, dim_in=None, ms=0):
    v = np.asarray(vec, dtype=float).ravel()
    return {"dim": int(v.size), "dim_in": int(dim_in) if dim_in is not None else None,
            "bars": binned(v), "stats": {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())},
            "ms": int(ms)}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return send_from_directory(BASE_DIR / "static", "index.html")


@app.get("/api/options")
def options():
    """What has been trained, so the page can disable combinations that do not exist yet."""
    selectors, classifiers = scan_models()
    metrics = load_metrics()
    return jsonify(
        extractors={k: {"finetuned": extractor_weights(k) is not None, "dim": FEATURE_DIM[k]} for k in EXTRACTORS},
        selectors=selectors,
        classifiers=classifiers,
        selected_dims=metrics.get("selected_dims", {}),
        metrics=metrics.get("results", {}),
        deep=metrics.get("deep", {}),
        dataset=metrics.get("dataset"),
        percent_range=[PERCENT_MIN, PERCENT_MAX],
        device=str(DEVICE),
        models_found=any(selectors.values()) and bool(classifiers),
    )


@app.post("/api/upload")
def upload():
    f = request.files.get("image")
    if f is None or not f.filename:
        raise PipelineError("No image received.")
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise PipelineError("Unsupported file type. Use PNG, JPG, BMP, TIFF or WebP.")
    try:
        img = Image.open(f.stream)
        img.load()
    except (UnidentifiedImageError, OSError):
        raise PipelineError("The file could not be read as an image.")
    if img.mode.startswith("I") or img.mode == "F":     # 16-bit / float scans -> 8-bit
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        img = Image.fromarray(arr.astype(np.uint8))
    img = img.convert("RGB")
    thumb = img.copy()
    thumb.thumbnail((640, 640))
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    preview = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return jsonify(token=new_session(img), preview=preview, width=img.width, height=img.height)


@app.post("/api/extract")
def extract():
    s, method, _ = _body(EXTRACTORS)
    t0 = time.perf_counter()
    try:
        X, source = run_extractor(method, s["img"])
    except PipelineError:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("extraction failed")
        raise PipelineError(f"Feature extraction failed: {e}", 500)
    s.update(features=X, extractor=method, selected=None, selector=None)
    out = summarize(X, ms=(time.perf_counter() - t0) * 1000)
    out["weights"] = source
    return jsonify(out)


@app.post("/api/select")
def select():
    s, method, _ = _body(SELECTORS)
    if s["features"] is None:
        raise PipelineError("Run feature extraction first.")
    ext = s["extractor"]
    path = selector_path(ext, method)
    if path is None:
        raise PipelineError(f"Model file not found: models/selectors/{ext}__{method}.pkl", 404)
    _, classifiers = scan_models()
    per_clf = classifiers.get(f"{ext}|{method}", {})
    if not per_clf:
        raise PipelineError(f"No classifiers found for {EXTRACTORS[ext]} + {SELECTORS[method]} "
                            f"(expected models/classifiers/{ext}__{method}__<classifier>__p<percent>.pkl). "
                            "Models from the old train_models.py have a different layout; run the new train_models.py.", 404)
    t0 = time.perf_counter()
    try:
        bundle = load_pickle(path)
        _require_bundle(bundle, path)
        Z, importance = apply_selector(bundle, s["features"])
    except PipelineError:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("selection failed")
        raise PipelineError(f"{SELECTORS[method]} could not be applied to the {EXTRACTORS[ext]} features "
                            f"({s['features'].shape[1]} values). Details: {e}", 500)
    s.update(selected=Z, selector=method)
    out = summarize(Z, dim_in=s["features"].shape[1], ms=(time.perf_counter() - t0) * 1000)
    out["importance"] = binned(importance, absolute=False)
    levels = sorted({lv for lvs in per_clf.values() for lv in lvs if PERCENT_MIN <= lv <= PERCENT_MAX})
    out["levels"] = levels or sorted({lv for lvs in per_clf.values() for lv in lvs})
    return jsonify(out)


@app.post("/api/classify")
def classify():
    s, method, data = _body(CLASSIFIERS)
    if s["selected"] is None:
        raise PipelineError("Run feature selection first.")
    try:
        requested = float(data.get("percent", PERCENT_MAX))
    except (TypeError, ValueError):
        raise PipelineError("Invalid percentage.")
    requested = max(PERCENT_MIN, min(PERCENT_MAX, requested))
    ext, sel = s["extractor"], s["selector"]
    _, classifiers = scan_models()
    levels = classifiers.get(f"{ext}|{sel}", {}).get(method, [])
    if not levels:
        raise PipelineError(f"Model file not found: models/classifiers/{ext}__{sel}__{method}__p<percent>.pkl", 404)
    level = min(levels, key=lambda lv: (abs(lv - requested), lv))       # nearest trained level
    path = classifier_path(ext, sel, method, level)
    t0 = time.perf_counter()
    try:
        bundle = load_pickle(path)
        _require_bundle(bundle, path)
        Z = s["selected"]
        total, k = Z.shape[1], int(bundle["k"])
        if k > total:
            raise PipelineError(f"{path.name} needs {k} features but the selector produced {total}. "
                                "Retrain so selector and classifier come from the same run.", 500)
        classes, proba, source = predict(bundle["model"], Z[:, :k])     # top-k ranked features
    except PipelineError:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("classification failed")
        raise PipelineError(f"{CLASSIFIERS[method]} could not classify {Z.shape[1]} features. Details: {e}", 500)

    top = int(np.argmax(proba))
    label, positive = class_info(classes[top])
    probs = sorted(({"label": class_info(c)[0], "p": float(p)} for c, p in zip(classes, proba)), key=lambda d: -d["p"])

    results = load_metrics().get("results", {})
    here = results.get(f"{ext}|{sel}|{method}|{level}")
    by_level = {str(lv): results[f"{ext}|{sel}|{method}|{lv}"]["accuracy"] for lv in levels
                if f"{ext}|{sel}|{method}|{lv}" in results}
    reliability = None
    if here:
        reliability = {key: here.get(key) for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc", "n_test")}
        reliability["by_level"] = by_level
    return jsonify(
        label=label, is_positive=bool(positive), confidence=float(proba[top]), probabilities=probs,
        probability_source=source, percent=level, requested_percent=requested,
        features_used=k, features_total=int(total), reliability=reliability,
        ms=int((time.perf_counter() - t0) * 1000))


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    log.info("Device: %s  |  open http://%s:%s", DEVICE, host, port)
    app.run(host=host, port=port, debug=False, threaded=True)
