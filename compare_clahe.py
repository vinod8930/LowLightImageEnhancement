import cv2
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model

# ===============================
# PATHS
# ===============================
BASE_PATH = r"C:\Users\vinod\Desktop\LowLightImageEnhancement\our485"
LOW_PATH  = os.path.join(BASE_PATH, "low")
HIGH_PATH = os.path.join(BASE_PATH, "high")

MODEL_PATH = "low_light_lol_model_saved"
IMG_SIZE = 256

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH, compile=False)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

psnr_clahe, ssim_clahe = [], []
psnr_cnn, ssim_cnn = [], []

files = sorted(os.listdir(LOW_PATH))[:30]

for f in files:
    low = cv2.imread(os.path.join(LOW_PATH, f))
    high = cv2.imread(os.path.join(HIGH_PATH, f))

    if low is None or high is None:
        continue

    low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
    high = cv2.cvtColor(high, cv2.COLOR_BGR2RGB)

    low_r = cv2.resize(low, (IMG_SIZE, IMG_SIZE))
    high_r = cv2.resize(high, (IMG_SIZE, IMG_SIZE))

    # ---------- CLAHE ----------
    lab = cv2.cvtColor(low_r, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    clahe_img = cv2.merge((l,a,b))
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2RGB)

    # ---------- CNN ----------
    inp = low_r / 255.0
    pred = model.predict(np.expand_dims(inp, axis=0))[0]
    cnn_img = (pred * 255).astype(np.uint8)

    # Metrics
    psnr_clahe.append(psnr(high_r, clahe_img, data_range=255))
    ssim_clahe.append(ssim(high_r, clahe_img, channel_axis=2, data_range=255))

    psnr_cnn.append(psnr(high_r, cnn_img, data_range=255))
    ssim_cnn.append(ssim(high_r, cnn_img, channel_axis=2, data_range=255))

# ===============================
# RESULTS
# ===============================
print("CLAHE  - Avg PSNR:", np.mean(psnr_clahe))
print("CLAHE  - Avg SSIM:", np.mean(ssim_clahe))
print("CNN    - Avg PSNR:", np.mean(psnr_cnn))
print("CNN    - Avg SSIM:", np.mean(ssim_cnn))
