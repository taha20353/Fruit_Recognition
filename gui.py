import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("fruit_model.h5")

# Class names (same order as training)
class_names = ['Apple Braeburn', 'Banana', 'Orange']

IMG_SIZE = (224, 224)

def predict_image(img_path):
    img = Image.open(img_path).resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    index = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return class_names[index], confidence

def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return

    # Show image
    img = Image.open(file_path)
    img.thumbnail((250, 250))
    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo)
    image_label.image = photo

    # Predict
    result, conf = predict_image(file_path)
    result_label.config(
        text=f"Prediction: {result}\nConfidence: {conf:.2f}%",
        fg="green"
    )

# Window
root = tk.Tk()
root.title("Fruit Classifier 🍎🍌🍊")
root.geometry("1920x1080")
root.resizable(False, False)

# Title
title = Label(root, text="Fruit Classification AI",
              font=("Arial", 16, "bold"))
title.pack(pady=10)

# Image display
image_label = Label(root)
image_label.pack(pady=10)

# Button
btn = Button(root, text="Select Image",
             font=("Arial", 12),
             command=open_image)
btn.pack(pady=10)

# Result
result_label = Label(root, text="",
                     font=("Arial", 12))
result_label.pack(pady=20)

# Run
root.mainloop()
