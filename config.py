"""
Global configuration for the Heart Sound Classification Project.
All hyperparameters and project constants are defined here.
"""

# ==========================================================
# DATA PARAMETERS
# ==========================================================

SAMPLE_RATE = 2000

WINDOW_DURATION = 2.52        # seconds

WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)

OVERLAP = 0.5

LOWCUT = 25

HIGHCUT = 400

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

N_MFCC = 13

N_MELS = 32

N_FFT = 256

HOP_LENGTH = 80

USE_DELTA = True

USE_DELTA_DELTA = True

NORMALIZE_FEATURES = True

# ==========================================================
# GAN PARAMETERS
# ==========================================================

LATENT_DIM = 100

GAN_BATCH_SIZE = 32

GAN_EPOCHS = 100

GAN_LEARNING_RATE = 0.0002

GAN_AUGMENT_RATIO = 0.60

# ==========================================================
# CLASSIFIER PARAMETERS
# ==========================================================

CLASSIFIER_BATCH_SIZE = 32

CLASSIFIER_EPOCHS = 40

CLASSIFIER_LEARNING_RATE = 0.001

DROPOUT = 0.30

EARLY_STOPPING_PATIENCE = 8

# ==========================================================
# TRAIN / VALIDATION / TEST
# ==========================================================

TRAIN_SPLIT = 0.70

VAL_SPLIT = 0.15

TEST_SPLIT = 0.15

SEED = 42

# ==========================================================
# EVALUATION
# ==========================================================

CLASSIFICATION_THRESHOLD = 0.45

SAVE_PLOTS = True