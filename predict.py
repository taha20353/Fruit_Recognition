import tensorflow as tf
import numpy as np
from keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("fruit_model.h5")

# Class names (MUST match training folder order)
class_names = ['Apple Braeburn', 'Banana', 'Orange']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

# Test image
predict_image("dataset/test/00000Apple1.jpg")
