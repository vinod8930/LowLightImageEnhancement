import os
import cv2
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

MODEL_PATH = "low_light_lol_model_saved"  # use saved model
IMG_SIZE = 256

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH, compile=False)

psnr_scores = []
ssim_scores = []

files = sorted(os.listdir(LOW_PATH))[:50]  # evaluate on 50 images

for f in files:
    low_img = cv2.imread(os.path.join(LOW_PATH, f))
    high_img = cv2.imread(os.path.join(HIGH_PATH, f))

    if low_img is None or high_img is None:
        continue

    # Convert to RGB
    low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
    high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)

    # Resize
    low_img_r = cv2.resize(low_img, (IMG_SIZE, IMG_SIZE))
    high_img_r = cv2.resize(high_img, (IMG_SIZE, IMG_SIZE))

    # Normalize
    inp = low_img_r / 255.0

    # Predict
    pred = model.predict(np.expand_dims(inp, axis=0))[0]
    pred = (pred * 255).astype(np.uint8)

    # Metrics
    psnr_val = psnr(high_img_r, pred, data_range=255)
    ssim_val = ssim(high_img_r, pred, channel_axis=2, data_range=255)

    psnr_scores.append(psnr_val)
    ssim_scores.append(ssim_val)

# ===============================
# RESULTS
# ===============================
print("Average PSNR:", np.mean(psnr_scores))
print("Average SSIM:", np.mean(ssim_scores))
