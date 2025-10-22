import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Limit TensorFlow thread usage for performance
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Load the trained MesoNet model
model = load_model("mesonet_model.h5")

def check_face_authenticity(frame, return_prob=False):
    face = cv2.resize(frame, (256, 256))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)

    pred = model.predict(face, batch_size=1)[0][0]

    if return_prob:
        return float(pred)

    return "REAL" if pred >= 0.5 else "FAKE"
