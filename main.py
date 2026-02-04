import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ===============================
# PATHS (YOUR DATASET)
# ===============================
BASE_PATH = r"C:\Users\vinod\Desktop\LowLightImageEnhancement\our485"
LOW_PATH  = os.path.join(BASE_PATH, "low")
HIGH_PATH = os.path.join(BASE_PATH, "high")

OUTPUT_PATH = r"C:\Users\vinod\Desktop\LowLightImageEnhancement\output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30

# ===============================
# MODEL
# ===============================
def build_cnn():
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = Conv2D(64, 3, padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    r = Conv2D(64, 3, padding="same")(x)
    r = BatchNormalization()(r)
    r = Activation("relu")(r)

    r = Conv2D(64, 3, padding="same")(r)
    r = BatchNormalization()(r)

    x = Add()([x, r])
    out = Conv2D(3, 3, padding="same", activation="sigmoid")(x)

    return Model(inp, out)

model = build_cnn()
model.compile(
    optimizer=Adam(1e-4),
    loss="mse"
)
model.summary()

# ===============================
# LOAD DATA
# ===============================
def load_dataset(low_dir, high_dir):
    X, Y = [], []
    files = sorted(os.listdir(low_dir))

    for f in tqdm(files, desc="Loading images"):
        low_img_path  = os.path.join(low_dir, f)
        high_img_path = os.path.join(high_dir, f)

        low_img  = cv2.imread(low_img_path)
        high_img = cv2.imread(high_img_path)

        if low_img is None or high_img is None:
            continue

        low_img  = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)

        low_img  = cv2.resize(low_img, (IMG_SIZE, IMG_SIZE))
        high_img = cv2.resize(high_img, (IMG_SIZE, IMG_SIZE))

        X.append(low_img / 255.0)
        Y.append(high_img / 255.0)

    return np.array(X), np.array(Y)

trainX, trainY = load_dataset(LOW_PATH, HIGH_PATH)

print("Dataset shape:", trainX.shape, trainY.shape)

# ===============================
# TRAIN
# ===============================
model.fit(
    trainX,
    trainY,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1
)

# ===============================
# SAVE MODEL
# ===============================
model.save("low_light_lol_model.h5")
print("Model saved")

# ===============================
# TEST + SAVE OUTPUTS
# ===============================
for f in os.listdir(LOW_PATH)[:10]:  # first 10 images
    img = cv2.imread(os.path.join(LOW_PATH, f))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    inp = img / 255.0

    pred = model.predict(np.expand_dims(inp, axis=0))[0]
    out = (pred * 255).astype(np.uint8)

    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUTPUT_PATH, f), out)

print("Enhanced images saved in:", OUTPUT_PATH)
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load old model
model = load_model("low_light_lol_model.h5", compile=False)

# Save in NEW format (SavedModel)
model.save("low_light_lol_model_saved")

print("Model re-saved successfully")
