import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import io


# ==========================================
# Custom Loss & Metric
# ==========================================

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    dice = (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        + smooth
    )

    return 1.0 - dice


def combined_loss(y_true, y_pred):
    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            y_true,
            y_pred
        )
    )

    dice = dice_loss(y_true, y_pred)

    return bce + dice


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_pred = tf.cast(
        y_pred > 0.5,
        tf.float32
    )

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    return (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        + smooth
    )


# ==========================================
# Load Model
# ==========================================

model = load_model(
    "seg model.keras",
    custom_objects={
        "combined_loss": combined_loss,
        "dice_loss": dice_loss,
        "dice_coefficient": dice_coefficient
    }
)


# ==========================================
# FastAPI
# ==========================================

app = FastAPI(
    title="Brain Tumor Segmentation API"
)


@app.get("/")
def home():
    return {
        "message": "Brain Tumor Segmentation API is running"
    }


# ==========================================
# Prediction Endpoint
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    # Resize exactly like training
    image = image.resize(
        (224, 224)
    )

    # Normalize
    image_array = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    # Add batch dimension
    input_image = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    pred = model.predict(
        input_image,
        verbose=0
    )

    # Create binary mask
    pred_mask = (
        pred[0, :, :, 0] > 0.5
    )

    # ======================================
    # Create Overlay
    # ======================================

    image_array_uint8 = (
        image_array * 255
    ).astype(np.uint8)

    overlay = image_array_uint8.copy()

    # Red tumor area
    overlay[pred_mask] = [
        255,
        0,
        0
    ]

    alpha = 0.4

    result = image_array_uint8.copy()

    result[pred_mask] = (
        alpha * overlay[pred_mask]
        + (1 - alpha)
        * image_array_uint8[pred_mask]
    ).astype(np.uint8)

    # Convert result to PNG
    result_image = Image.fromarray(
        result
    )

    output = io.BytesIO()

    result_image.save(
        output,
        format="PNG"
    )

    output.seek(0)

    return StreamingResponse(
        output,
        media_type="image/png"
    )