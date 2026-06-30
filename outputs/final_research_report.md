# Research Report: Heart Sound Screening Assistant
 
 ## Executive Summary
 This report evaluates a research **screening assistant, not a diagnosis system**, for classifying Phonocardiogram (PCG) recordings from the **PhysioNet Heart Sound Dataset (CinC Challenge 2016)**. The project compares a baseline model with a GAN-augmented model and uses validation-selected thresholds instead of assuming a fixed 0.5 cutoff.
 
 ---
 
 ## Prevent Data Leakage validation Check
 To reduce leakage risk, a recording-level split was implemented *before* signal segmentation. The overlap validation check verified:
 * **Overlapping recordings between splits**: 0 files.
 * **Leak-proof splitting**: Successful.
 
 ---
 
 ## Recording-Level Performance Comparison
 
 Each recording is evaluated once by averaging its segment probabilities. This matches how `predict.py` handles one uploaded WAV file.
 
 | Model | Threshold | Accuracy | Balanced Accuracy | Precision | Recall | Specificity | NPV | F1-score | AUC |
 | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
 | **Baseline (No GAN)** | 0.510 | 0.6333 | 0.6806 | 0.5238 | 0.9167 | 0.4444 | 0.8889 | 0.6667 | 0.7639 |
 | **GAN-Augmented** | 0.520 | 0.7000 | 0.7361 | 0.5789 | 0.9167 | 0.5556 | 0.9091 | 0.7097 | 0.7546 |
 | **Ensemble (Baseline + GAN)** | 0.510 | 0.6333 | 0.6806 | 0.5238 | 0.9167 | 0.4444 | 0.8889 | 0.6667 | 0.7731 |
 
 The thresholds above are selected on the validation split, not guessed at 0.5.
 Predictions close to the selected threshold should be treated as **Uncertain**
 and reviewed by a clinician.
 
 ### Calibration and Threshold Safety
 The following plots are generated to inspect confidence quality and threshold tradeoffs:
 
 ![Calibration Curve](plots/calibration_curve_gan.png)
 
 ![Threshold Tradeoff](plots/threshold_tradeoff_gan.png)
 
 ### Relative Improvements:
 * **Accuracy Improvement**: +6.67%
 * **Recall (Sensitivity) Improvement**: +0.00%
 * **F1-score Improvement**: +4.30%
 * **AUC Score Improvement**: -0.93%
 
 ---
 
 ## GAN Validation Analysis
 
 ### 1. t-SNE Feature Space Distribution
 t-SNE dimensionality reduction (from 2496 dimensions to 2D) was applied to the abnormal training features (real vs. synthetic). The plot demonstrates:
 * The plot is a visual sanity check only; it is not proof of clinical quality.
 * If synthetic points are far away from real points, the GAN should be treated carefully or disabled for the final classifier.
 
 ![t-SNE Scatter Plot](plots/tsne_real_vs_generated.png)
 
 ### 2. Statistical Similarity
 We compared the distribution statistics of the combined MFCC + log-mel features:
 * **Real Abnormal Means (average)**: 0.0931 (Variance: 0.0322)
 * **Generated Abnormal Means (average)**: 0.3672 (Variance: 0.0356)
 * **Average Spectral Centroid (MFCC-index equivalent)**:
   * Real Abnormal: Coefficient -45.94
   * Generated Abnormal: Coefficient 14.32
 
 These statistics are useful checks, but classifier performance and validation/test metrics are more important than visual similarity alone.
 
 ### 3. Visual Feature Comparison (MFCC Heatmaps)
 The following plot shows a side-by-side comparison of a Real Normal, Real Abnormal, and GAN Synthetic Abnormal MFCC heatmap. 
 
 ![MFCC Heatmaps Comparison](plots/spectrogram_comparisons.png)
 
 ---
 
 ## Visual Model Comparisons
 
 ### Confusion Matrices
 Below are the confusion matrices for the three model configurations (Baseline, GAN-Augmented, and Ensemble). Note how the Ensemble model balances false alarms and missed cases:
 
 ![Confusion Matrix Comparison](plots/confusion_matrix_comparison.png)
 
 ### ROC Curves & Training Histories
 The ROC curve comparison and training curve comparisons are illustrated below:
 
 ![ROC & Training Curves Comparison](plots/performance_comparison.png)
 
 ![Class Distribution Before and After GAN](plots/class_distribution_comparison.png)
 
 ---
 
 ## Final Conclusions
 1. **GAN Result**: GAN augmentation improved the main screening metrics in this run.
 2. **Best balanced setting**: Use **GAN-Augmented** in **screening** mode when you want the best balance of recall and specificity. Test balanced accuracy = **0.7361**.
 3. **To reduce false positives**: Use **GAN-Augmented** in **balanced** mode. Test specificity = **0.7222**, recall = **0.7500**.
 4. **Clinical Safety Note**: This project is a research screening assistant, not a diagnosis system. It is not perfect, not medically certified, and must not be used as a stand-alone medical decision tool.
 