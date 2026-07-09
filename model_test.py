import tensorflow as tf
import tf_keras

model = tf_keras.models.load_model(
    "keras_model.h5",
    compile=False
)

print("Model loaded successfully.")