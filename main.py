import tensorflow as tf
import tf_keras as keras

print("TensorFlow Version:", tf.__version__)
print("Loading model...")

model = keras.models.load_model("keras_model.h5", compile=False)

print("✅ Model loaded successfully!")

print("\n========== MODEL SUMMARY ==========\n")
model.summary()
