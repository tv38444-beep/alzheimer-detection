# AlzDetect - Alzheimer's detection from brain scans

Scan -> CNN feature extraction -> feature selection -> classifier -> *Alzheimer's / Non-Alzheimer's*

```
Extractor : ResNet-50 | EfficientNet-B0 | ConvNeXt-tiny | DenseNet-121   (transfer learning)
Selector  : PCA | mRMR | PSO | RFE      -> keeps 600-900 features, ranked best-first
Budget    : slider 10%-60%              -> only the TOP k% of those features go to the classifier
Classifier: SVM | Random Forest | KNN | XGBoost
```

## Run the web app (VS Code)

```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py                    # open http://127.0.0.1:5000
python check_models.py           # which model files are present
```

## Train the models on YOUR data (`train_models.py`, run in Google Colab with GPU)

```python
!pip install -q torch torchvision scikit-learn xgboost pandas joblib
!python train_models.py --data /content/your_dataset
```
`--quick` = fast trial run, `--only efficientnet_b0` = one extractor, `--no-reuse` = re-fine-tune the CNNs.
Then copy the generated **`models/`** folder into this project (next to `app.py`).

Dataset layout: one sub-folder per class (e.g. `NonDemented`, `VeryMildDemented`, `MildDemented`,
`ModerateDemented`). `NonDemented` -> label 0, everything else -> label 1 (Alzheimer's).
Check the folder -> label table the script prints. Edit `CLASS_MAP` if your names differ.

### Why accuracy holds when you use only 10-60% of the features
* Selectors keep **600-900 features** (`SELECT_FRACTION=0.8`, `SELECT_MAX=900`: 1280->900, 2048->900,
  1024->819, 768->614) and rank them best-first.
* The slider picks the **top k%** of the ranked features. A separate classifier is trained for each
  level (10, 20, ... 60%) on exactly those features, so the model has seen that input size in training.
  (Zeroing features out of a classifier trained on all of them loses accuracy - that was the old design.)
* Every (extractor, selector, classifier, level) is scored on a **held-out test split**; results go to
  `results.csv` and `models/metrics.json`, and the web page shows them next to each prediction.

### Honest-accuracy safeguards
* **Split by original image, not by file.** Augmented copies (`aug_00400_mild_13.jpg`) of one original
  always stay together in train / val / test and inside cross-validation (`GroupKFold`).
* The test split is never used for fitting, tuning or choosing epochs.
* If you re-use fine-tuned weights (`REUSE_FINETUNED_WEIGHTS=True`) the dataset and `RANDOM_STATE`
  must be unchanged, so the CNN never saw your current test images. The split is saved to
  `models/split.csv` and re-used automatically on later runs.

### Files produced
```
models/extractors/<extractor>.pth
models/selectors/<extractor>__<selector>.pkl                      (16 files)
models/classifiers/<extractor>__<selector>__<classifier>__p<percent>.pkl   (16 x levels)
models/metrics.json   models/split.csv   results.csv
```
Models from the previous version have a different format and are rejected with a clear message:
re-run the new `train_models.py` (with `REUSE_FINETUNED_WEIGHTS=True` it keeps your fine-tuned CNNs).

### Tuning knobs (top of `train_models.py`)
`PERCENT_LEVELS` (e.g. `range(10, 61, 5)` for a finer slider), `SELECT_MAX`, `SELECT_FRACTION`,
`UNFREEZE` ("all" = best accuracy, "last_block" = faster), `FINE_TUNE_EPOCHS`, `SVM_MAX_ROWS`.

## Notes
* Input: PNG/JPG/BMP/TIFF/WebP. The dataset you trained on decides what the model understands (MRI
  slices in -> MRI model). Use the same kind of scan at prediction time.
* Research prototype, not a medical device.
