# ---- Data ----
SAMPLE_RATE = 4000
WINDOW_SIZE = 10.0    # seconds
WINDOW_STRIDE = 5.0   # seconds (sliding mode)
WINDOW_MODE = 'fixed' # 'fixed' or 'sliding'
N_MELS = 40

# ---- STFT ----
N_FFT = 512
HOP_LENGTH = 40
WIN_LENGTH = 100

# ---- Model Architecture ----
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
KERNEL_SIZE = 31
EXPANSION_FACTOR = 4
DROPOUT = 0.3
DROP_PATH_RATE = 0.1

# ---- ALE / MRAB ----
MU_MODE = 'scalar'   # 'fixed' | 'scalar'
MU_INIT = 0.05

# ---- Training ----
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-4
PATIENCE = 10
SEED = 42

# ---- Augmentation ----
USE_AUGMENTATION = True
MIXUP_ALPHA = 0.4
FILTERAUG_PROB = 0.5

# ---- ALE Separate Training ----
USE_SEPARATE_ALE_TRAINING = False
ALE_LR = 1e-4
ALE_UPDATE_FREQ = 1

# ---- Output ----
OUTPUT_DIR = "./results"
