"""Shows which trained model files exist in ./models (new layout with percent levels)."""
import json
from pathlib import Path

M = Path(__file__).resolve().parent / "models"
EXT = ["resnet50", "efficientnet_b0", "convnext_tiny", "densenet121"]
SEL = ["pca", "mrmr", "pso", "rfe"]
CLF = ["svm", "random_forest", "knn", "xgboost"]


def has(folder, stem):
    return any((M / folder / f"{stem}{e}").is_file() for e in (".pkl", ".joblib", ".pth"))


metrics = {}
if (M / "metrics.json").is_file():
    metrics = json.loads((M / "metrics.json").read_text(encoding="utf-8"))
levels = metrics.get("levels", [10, 20, 30, 40, 50, 60])
print(f"percent levels expected: {levels}\n")
for e in EXT:
    w = has("extractors", e)
    print(f"{e}  (fine-tuned weights: {'yes' if w else 'NO - ImageNet weights will be used'})")
    for s in SEL:
        n = metrics.get("selected_dims", {}).get(f"{e}|{s}")
        sel_ok = has("selectors", f"{e}__{s}")
        row = []
        for c in CLF:
            got = [lv for lv in levels if has("classifiers", f"{e}__{s}__{c}__p{lv}")]
            row.append(f"{c}:{len(got)}/{len(levels)}")
        print(f"  {s:5s} selector {'ok     ' if sel_ok else 'MISSING'}"
              f"{(' (' + str(n) + ' features)') if n else ''} | {'  '.join(row)}")
print("\nmetrics.json:", "found" if metrics else "MISSING (the page cannot show test accuracy)")
