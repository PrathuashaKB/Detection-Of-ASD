import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('asd_model.h5')

# Set the image size
SIZE = 100  # Update this based on your model's input shape

# Create the Streamlit app
st.title('ASD Detection Web App')
st.write('Upload an image for ASD detection')

# Get user input
uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((SIZE, SIZE))
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values

    # Make predictions
    prediction = model.predict(np.expand_dims(image_array, axis=0))
    predicted_class = int(prediction > 0.5)

    if predicted_class == 0:
        st.write('Prediction: ASD (Positive)')
    else:
        st.write('Prediction: Non-ASD (Negative)')

