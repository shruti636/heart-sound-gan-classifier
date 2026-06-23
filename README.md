# GAN-Augmented Heart Sound Signal Classification Study

An end-to-end Deep Learning research study evaluating the effectiveness of Generative Adversarial Networks (GANs) to address class imbalance in Phonocardiogram (PCG) heart sound classification. 

This project uses the official **PhysioNet Heart Sound Dataset (CinC Challenge 2016)**, extracts Mel-Frequency Cepstral Coefficients (MFCCs), implements a strict **leak-proof recording-level split**, and compares a Baseline (No-GAN) model with a GAN-Augmented model. Validation includes **t-SNE dimensionality reduction** to verify the realism of generated sequences.

---

## Data Validation & Leakage Prevention Pipeline

In audio sequence classification, segmenting signals before train-test splitting introduces **data leakage** because segments from the same recording share extreme temporal correlation, artificially inflating test metrics. 

To ensure scientific validity, our preprocessing pipeline implements a **recording-level stratified split**:
1.  **Audio Splitting**: The 100 raw `.wav` recordings are stratified and split into Train (69 files), Validation (16 files), and Test (15 files) *first*.
2.  **Overlap Auditing**: An automated assertion checks that `Train_Set ∩ Test_Set = ∅` and writes file manifests to `train_source_files.txt` and `test_source_files.txt`.
3.  **Split Segmentation**: Preprocessing, Butterworth bandpass filtering, and 13-coefficient MFCC extraction are executed on each split independently, yielding features of shape `(batch, 64, 13)`.

```
Heart Sound recordings (.wav)
→ Stratified Recording-Level Split (70/15/15)
→ Butterworth Filtering (25-400Hz) & Normalization
→ Fixed-Length Segmentation (2.52s windows, overlap 1s)
→ 13-coefficient MFCC Extraction
→ X_train.npy, X_val.npy, X_test.npy (No Leakage!)
```

---

## Network Architectures

*   **1D DCGAN**: Trained on real abnormal training MFCCs only. The Generator uses `Conv1DTranspose` layers to map a 100-dimensional noise vector $z$ to a synthetic MFCC matrix of shape `(64, 13)`. The Discriminator uses `Conv1D` layers to evaluate features as real vs. fake.
*   **CNN/LSTM Classifier**: A hybrid model consisting of three `Conv1D` blocks for local temporal feature extraction, followed by a `Bidirectional LSTM` layer for global sequence modeling, ending in a Sigmoid Dense unit.

---

## Algorithms and Mathematical Explanations

### 1. MFCC Feature Extraction
MFCCs model the human auditory system by applying the following steps:
1.  **FFT Power Spectrum**: Convert each Hanning-windowed frame of segmented audio to the frequency domain:
    \[X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi}{N} kn}\]
2.  **Mel-Scale Filtering**: Filter the power spectrum using a triangular filterbank spaced logarithmically on the Mel scale:
    \[m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)\]
3.  **Discrete Cosine Transform (DCT)**: Decorrelate the log Mel filterbank energies $S_k$ to obtain the cepstral coefficients:
    \[c(n) = \sum_{k=1}^K S_k \cos\left(n \left(k - \frac{1}{2}\right) \frac{\pi}{K}\right)\]
    We select the first **13 coefficients** ($n=1,\dots,13$) to capture spectral envelopes.

### 2. t-SNE Feature Space Projection
To validate that the GAN synthesized realistic abnormal samples, we use **t-Distributed Stochastic Neighbor Embedding (t-SNE)**. t-SNE maps high-dimensional flattened MFCC features ($D = 64 \times 13 = 832$) to a 2D space by minimizing the Kullback-Leibler (KL) divergence between probability distributions that represent similarities:
\[KL(P || Q) = \sum_{i} \sum_{j} p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}\]
This projects the real and generated abnormal feature spaces to show how closely the synthetic data distribution overlaps with the real data distribution.

---

## Step-by-Step Execution Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Real PhysioNet Dataset
Download the first 60 Normal and 40 Abnormal `.wav` files and REFERENCE annotations from PhysioNet training-a dataset:
```bash
python download_physionet.py
```
This writes the recordings to `data/raw/` and creates the label file `data/raw/reference.csv`.

### Step 3: Run Preprocessing & Feature Extraction
Perform recording-level splitting, check for leakage, segment the files, and save the splits:
```bash
python preprocessing.py
```
Outputs separate train, val, and test arrays in `data/processed/`.

### Step 4: Train the 1D GAN Model
Train the GAN strictly on the real abnormal training segments:
```bash
python train_gan.py --epochs 50 --batch_size 32
```
Saves the generator checkpoint to `models/gan_generator.keras` and output plots to `outputs/mfcc_plots/`.

### Step 5: Train Both Classifiers
Train the Baseline (No-GAN) and the GAN-Augmented classifiers:
```bash
python train_classifier.py --epochs 20 --batch_size 16
```
Saves baseline weights to `models/cnn_classifier_nogan.keras` and GAN-augmented weights to `models/cnn_classifier_gan.keras`.

### Step 6: Run Comparative Study & Evaluation
Run predictions, generate ROC and Confusion Matrix comparisons, run t-SNE projection, and write the research report:
```bash
python evaluation.py
```
This generates:
*   `outputs/plots/performance_comparison.png` (ROC and loss curves)
*   `outputs/plots/confusion_matrix_comparison.png` (side-by-side matrices)
*   `outputs/plots/class_distribution_comparison.png` (balancing bar chart)
*   `outputs/plots/tsne_real_vs_generated.png` (t-SNE projection space)
*   `outputs/gan_vs_no_gan_results.csv` (summary table)
*   `outputs/final_research_report.md` (scientific report)
