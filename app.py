
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("/content/drive/MyDrive/model_multiclass.h5")

# Define class labels
class_labels = ["class1", "class2", "class3"]  # Replace with your actual class labels

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img.astype("float") / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except cv2.error as e:
        st.error(f"Error processing image: {e}")
        return None

def make_prediction(image_path):
    if image_path:
        preprocessed_img = preprocess_image(image_path)
        if preprocessed_img is not None:
            prediction = model.predict(preprocessed_img)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_labels[predicted_class_index]
            confidence = prediction[0][predicted_class_index]

            st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2%})")
            st.image(image_path, caption="Uploaded Image", use_column_width=True)
        else:
            st.error("Failed to process the image.")
    else:
        st.warning("Please upload an image.")

st.title("Image Classification App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if st.button("Make Prediction"):
    make_prediction(uploaded_file)
