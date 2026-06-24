# Beginner-Friendly GAN-Augmented Heart Sound Classification

An educational, end-to-end Deep Learning project that demonstrates how to process heart sound signals, extract robust features, train a Generative Adversarial Network (GAN) to balance classes, and classify heart sounds using a hybrid neural network.

This project is built using a simple, clean, and highly readable codebase designed for beginners learning Signal Processing, Machine Learning, and Generative AI.

---

## Educational Concepts

### 1. What are Heart Sounds?
Heart sounds are acoustic waves produced by the opening and closing of heart valves during the cardiac cycle. 
* **Normal Heart Sounds**: Consist of two main sounds, often described as "lub" (first heart sound, S1) and "dub" (second heart sound, S2). S1 represents the closure of tricuspid and mitral valves; S2 represents the closure of aortic and pulmonary valves.
* **Abnormal Heart Sounds (Murmurs)**: Auditory murmurs or clicks occur due to turbulent blood flow through diseased, narrowed, or leaking heart valves. These waves are captured as **Phonocardiogram (PCG)** recordings (saved as `.wav` audio files).

### 2. Feature Extraction: MFCC, Delta, and Delta-Delta
A raw audio wave is highly redundant and difficult for deep networks to learn directly. Instead, we extract **Mel-Frequency Cepstral Coefficients (MFCCs)**:
* **MFCC (13 Coefficients)**: Represents the short-term power spectrum of a sound, simulating how humans perceive frequencies logarithmically using the Mel scale.
* **Delta MFCC (13 Coefficients)**: The first-order derivative of the MFCCs over time, representing the velocity (rate of change) of the spectral characteristics.
* **Delta-Delta MFCC (13 Coefficients)**: The second-order derivative, representing the acceleration of the spectral characteristics.

By stacking MFCCs, Delta MFCCs, and Delta-Delta MFCCs, we get a feature matrix of shape `(64, 39)` (64 time frames, 39 features). This captures both the static frequencies and the dynamic transitions of heart murmurs!

### 3. Why Use a GAN for Dataset Balancing?
In medical datasets, normal cases are usually far more common than abnormal cases. Training a classifier on unbalanced data leads to **class collapse**, where the model simply predicts "Normal" for everything and fails to detect abnormalities.
* To solve this, we train a **1D DCGAN (Deep Convolutional GAN)** strictly on the minority class (Abnormal) segments.
* The **Generator** learns to create realistic, synthetic abnormal feature matrices of shape `(64, 39)` from random noise.
* The **Discriminator** learns to distinguish real abnormal matrices from synthetic ones.
* Once trained, we use the generator to synthesize new abnormal samples until we have a 50/50 balanced dataset.

### 4. Why Use a Conv1D + BiLSTM Classifier?
Our hybrid neural network uses:
1. **Conv1D Layers**: Extract local patterns and shapes from the 39-dimensional feature matrix.
2. **Bidirectional LSTM (BiLSTM) Layer**: Models the temporal sequence. LSTMs remember long-term dependencies, and the bidirectional wrapper allows the model to read the heart sound sequence both forward (left-to-right) and backward (right-to-left) for maximum context.

---

## Project Pipeline

```
PhysioNet Heart Sound Dataset (.wav)
        ↓
Recording-Level Split (Leakage Prevention)
        ↓
Butterworth Bandpass Filter (25Hz–400Hz) & Normalization
        ↓
Fixed-Length Windowing (2.52s segments)
        ↓
MFCC + Delta + Delta-Delta Extraction (64 × 39 matrices)
        ↓
1D DCGAN Augmentation (trained on Abnormal features)
        ↓
Dataset Balancing (synthesizing abnormal samples)
        ↓
Hybrid Conv1D + BiLSTM Classifier
        ↓
Evaluation & Prediction System
```

---

## Step-by-Step Execution Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download the PhysioNet Dataset
Download 117 Normal and 80 Abnormal recordings from the official challenge training set:
```bash
python download_physionet.py
```

### Step 3: Run Preprocessing & Feature Extraction
Split recordings into leak-proof Train/Val/Test groups, filter out noise, segment the signals, and extract the 39-feature matrices:
```bash
python preprocessing.py
```
This saves preprocessed arrays (`X_train.npy`, `y_train.npy`, etc.) to `data/processed/`.

### Step 4: Train the 1D GAN
Train the generator strictly on the abnormal training segments to learn murmur characteristics:
```bash
python train_gan.py --epochs 50 --batch_size 32
```
Saves the trained models to the `models/` directory and outputs diagnostic plots to `outputs/plots/gan_loss.png`.

### Step 5: Train Both Classifiers (Comparison Study)
Train a Baseline classifier (unbalanced real data) and a GAN-augmented classifier (balanced using the generator):
```bash
python train_classifier.py --epochs 20 --batch_size 16
```

### Step 6: Evaluate & Compare Performance
Calculate predictions, generate training curves, t-SNE projections, and confusion matrices:
```bash
python evaluation.py
```
This compiles the quantitative results to `outputs/gan_vs_no_gan_results.csv` and compiles the scientific report to `outputs/final_research_report.md`.

---

## Running Inference and the Streamlit Dashboard

### 1. Test Single WAV Files via CLI
Use the standalone prediction script to classify any audio recording:
```bash
python predict.py --wav data/raw/a0007.wav
```
This output-only pipeline runs filtering, MFCC extraction, and feeds the sequence into the model, printing:
```
==================================================
         HEART SOUND INFERENCE RESULT
==================================================
File Path:  data/raw/a0007.wav
Prediction: Normal
Confidence: 89.24%
Segments:   16
==================================================
```

### 2. Launch the Streamlit Dashboard
To run a clean, interactive user interface where you can upload and test WAV files:
```bash
streamlit run app.py
```
The dashboard features:
* **Audio Player**: Listen to the uploaded heart sound.
* **Raw Waveform Plot**: Visualizes the amplitude of the signal.
* **Feature Heatmap**: Renders the 39 MFCC + Delta + Delta-Delta features.
* **Model Inference**: Displays the predicted class and confidence score.
