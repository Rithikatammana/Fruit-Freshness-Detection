import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("fruit_model.h5", compile=False)

# class labels
class_names = [
    "Fresh Apple",
    "Fresh Banana",
    "Fresh Orange",
    "Rotten Apple",
    "Rotten Banana",
    "Rotten Orange"
]

st.title("🍎 Fruit Freshness Detection using AI")
st.write("This application uses a Deep Learning model (MobileNetV2) to classify whether a fruit is Fresh or Rotten.")

file = st.file_uploader("Upload Fruit Image", type=["jpg","png","jpeg"])

if file:
    image = Image.open(file).resize((224,224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    result = class_names[class_index]

    st.success(f"Prediction: {result}")
    confidence = np.max(prediction)*100
    st.write(f"Confidence: {confidence:.2f}%")
