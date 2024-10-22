import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image

# Function to load the model
@st.cache(allow_output_mutation=True)  # Caching the model to avoid reloading on every run
def load_cnn_model():
    model = load_model('64x3-CNN.model')  # Load the model
    return model

# Function to predict the class of the uploaded image
def predict_class(model, image):
    # Resize and normalize the image
    RGBImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (224, 224))  # Adjust the size to fit model input
    image_array = np.array(RGBImg) / 255.0

    # Make prediction
    prediction = model.predict(np.array([image_array]))
    result = np.argmax(prediction, axis=1)

    return result

# Streamlit app
def main():
    st.title("Diabetic Retinopathy Detection")

    # Load the model
    model = load_cnn_model()

    # Upload image section
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image
        st.image(opencv_image, channels="BGR")

        # Make predictions
        prediction = predict_class(model, opencv_image)

        # Display result
        if prediction == 1:
            st.success("Prediction: No DR (No Diabetic Retinopathy)")
        else:
            st.error("Prediction: DR (Diabetic Retinopathy)")

if __name__ == '__main__':
    main()
