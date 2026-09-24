# scripts/config.py
import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ISAAC_DATASET_DIR = PROJECT_ROOT / "isaac_dataset"
ML_DATASET_PATH = PROJECT_ROOT / "ml_dataset_v2.h5"
MODELS_DIR = PROJECT_ROOT / "models"
PROP_LSTM_PATH = MODELS_DIR / "prop_lstm.pth"
CNN_MODEL_PATH = MODELS_DIR / "cnn_multi.pth"
USD_PATH = Path(os.getenv("QUAD_USD_PATH", PROJECT_ROOT / "quad.usd"))

# window / variables (paper values)
WINDOW = int(os.getenv("WINDOW", 100))
WINDOW_STEP = int(os.getenv("WINDOW_STEP", 10))
INPUT_VARS = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","rpm1","rpm2","rpm3","rpm4"]
SEED = int(os.getenv("SEED", 42))

# create directories if missing
for d in (ISAAC_DATASET_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
