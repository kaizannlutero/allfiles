from picamera2 import Picamera2
import cv2
import numpy as np
import time
import tensorflow as tf
import tf_keras as keras

model = keras.models.load_model("keras_model.h5", compile=False)

with open("labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (2028, 1520), "format": "BGR888"}
)
picam2.configure(config)
picam2.start()

time.sleep(2)

print("Mold detector started. Press Q to quit.")

try:
    while True:
        frame = picam2.capture_array()

        img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        img = (img / 127.5) - 1

        prediction = model.predict(img, verbose=0)
        index = np.argmax(prediction)
        label = labels[index]
        confidence = prediction[0][index] * 100

        display = cv2.resize(frame, (960, 720))
        text = f"{label}: {confidence:.2f}%"
        cv2.putText(display, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

        cv2.imshow("Mold Detection", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()
