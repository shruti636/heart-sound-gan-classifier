# GAN-Augmented Heart Sound Classification

Beginner-friendly deep learning project for classifying phonocardiogram (PCG)
heart sounds as **Normal** or **Abnormal**.

The project uses:

- Audio preprocessing with a Butterworth bandpass filter.
- MFCC + delta + delta-delta features plus log-mel spectrogram features.
- A small DCGAN to generate minority-class abnormal feature samples.
- A compact Conv1D classifier for final prediction.

## Why This Version Is Beginner Friendly

Earlier versions used attention blocks, WGAN-GP, custom losses, reconstruction
loops, and deeper classifiers. Those ideas can be useful, but they make the
project harder to learn from.

This version keeps the important idea:

```text
raw WAV -> clean audio -> segments -> MFCC + log-mel features -> GAN balancing -> CNN-BiGRU classifier
```

The code is shorter, faster to train, and easier to debug.

## Project Files

| File | Purpose |
| --- | --- |
| `download_physionet.py` | Downloads the sample PhysioNet heart sound data. |
| `preprocessing.py` | Cleans audio, segments it, extracts `(64, 71)` MFCC + log-mel features. |
| `gan_model.py` | Small generator and discriminator models. |
| `train_gan.py` | Trains the GAN on abnormal training samples. |
| `cnn_classifier.py` | Compact CNN + BiGRU classifier. |
| `train_classifier.py` | Trains baseline and GAN-augmented classifiers. |
| `evaluation.py` | Compares baseline vs GAN-augmented performance. |
| `cross_validate_classifier.py` | Runs recording-level cross-validation. |
| `predict.py` | Predicts Normal/Abnormal for one WAV file. |
| `app.py` | Streamlit dashboard. |
| `project_walkthrough.ipynb` | Beginner notebook explaining the full project. |

## Setup

```bash
pip install -r requirements.txt
```

## Run the Full Pipeline

### 1. Download data

```bash
python download_physionet.py
```

### 2. Preprocess WAV files

```bash
python preprocessing.py
```

This creates arrays like `data/processed/X_train.npy` and `y_train.npy`.

### 3. Train the GAN

```bash
python train_gan.py --epochs 50 --batch_size 32
```

The GAN learns only from abnormal feature samples and saves:

- `models/gan_generator.keras`
- `models/gan_discriminator.keras`
- `outputs/plots/gan_loss.png`

### 4. Train classifiers

```bash
python train_classifier.py --epochs 35 --batch_size 16 --augmentation_ratio 0.5
```

This trains:

- `models/cnn_classifier_nogan.keras`: baseline model.
- `models/cnn_classifier_gan.keras`: model trained with GAN-balanced data.

The default `augmentation_ratio=0.5` intentionally avoids adding too many GAN
samples, which can reduce false positives compared with forcing perfect class
balance. Synthetic samples are filtered by the GAN discriminator when available.

### 5. Evaluate

```bash
python evaluation.py
```

Results are saved in `outputs/`, including CSV metrics, threshold tradeoff plots,
and confidence calibration plots.

### 6. Optional recording-level cross-validation

```bash
python cross_validate_classifier.py --folds 3 --epochs 10
```

This checks whether results are stable across multiple recording-level splits.

### 7. Predict one audio file

```bash
python predict.py --wav data/raw/a0007.wav
```

### 8. Launch the app

```bash
streamlit run app.py
```

## Accuracy Notes

This project is a research and learning project, not a certified medical
device. Patients should not rely on the raw model output as a diagnosis.

For medical-style ML projects, the "best" model is not always the one with high
accuracy or high recall. You should also check:

- **Recall**: catches abnormal cases.
- **Specificity**: avoids incorrectly alarming normal cases.
- **Precision**: avoids false alarms.
- **NPV**: how reliable a normal prediction is.
- **AUC**: ranking quality across thresholds.
- **Confusion matrix**: actual mistakes by class.

After running `python evaluation.py`, the project saves validation-selected
decision thresholds to `models/model_thresholds.json`. The prediction script and
dashboard use those thresholds and return **Uncertain** when a recording is too
close to the threshold. Those uncertain cases should be reviewed by a clinician.

Prediction supports two operating modes:

```bash
python predict.py --wav data/raw/a0007.wav --mode screening
python predict.py --wav data/raw/a0007.wav --mode balanced
python predict.py --wav data/raw/a0007.wav --mode specificity
```

- `screening`: favors abnormal recall, so it is less likely to miss abnormal
  cases but can create many false alarms.
- `balanced`: uses a more even validation tradeoff, so it reduces false alarms
  but may miss more abnormal cases.
- `specificity`: reduces false positives by requiring stronger abnormal
  evidence, but may miss more abnormal cases.

GAN augmentation can help when abnormal samples are rare, but it must always be
validated against a held-out test split and real-world data before any clinical
use.
