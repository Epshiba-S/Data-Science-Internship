import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. Page Title and Layout Config
st.set_page_config(page_title="Intel Scene Classifier", page_icon="🏔️", layout="centered")

# 2. Inject Custom CSS for Background Styling (Fixed Argument!)
st.markdown(
    """
    <style>
    /* Changes the main background color of the app */
    .stApp {
        background-color: #1E293B; /* Dark slate blue/gray background */
        color: #F8FAFC; /* Crisp off-white text color */
    }
    
    /* Styles the image upload box */
    .stFileUploader {
        background-color: #334155;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #64748B;
    }
    
    /* Target regular text labels to remain highly visible */
    label, p, .stMarkdown {
        color: #F8FAFC !important;
    }
    
    /* Styled container card for results */
    .result-card {
        background-color: #0F172A;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #38BDF8;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📸 Intel Image Scene Classifier")
st.write("Upload a scenery photo below, and your newly trained neural network will process it instantly.")

# 3. Load the model safely using your confirmed absolute directory path
@st.cache_resource
def load_trained_cnn():
    model_path = r"C:\Users\epshi\Downloads\Intel_image\my_model.h5"
    return tf.keras.models.load_model(model_path)

try:
    model = load_trained_cnn()
    st.sidebar.success("🤖 Neural Network Model Connected!")
except Exception as e:
    st.sidebar.error(f"⚠️ Model initialization error: {e}")

# 4. Intellectual Dataset categories mapped in standard alphabetical order
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# 5. User Interface: File Drop Zone
uploaded_file = st.file_uploader("Select or drop your image format (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Preview Image', use_container_width=True)
    
    with st.spinner("Processing image properties..."):
        # Image Transformations matching your training parameters perfectly
        resized_img = image.resize((150, 150))
        img_array = np.array(resized_img, dtype=np.float32)
        
        # Security Guard: Ensure exact 3-channel RGB image format layout
        if img_array.shape[-1] != 3:
            img_array = img_array[:, :, :3]
            
        # Match explicit training optimization division scaling factor [0.0 - 1.0]
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict class values
        predictions = model.predict(img_array)
        score = predictions[0]  # The model outputs direct softmax distributions
        
        predicted_index = np.argmax(score)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = score[predicted_index] * 100

    # 6. HTML/CSS Styled Results Box
    st.markdown(
        f"""
        <div class="result-card">
            <h3 style='margin:0; color:#38BDF8;'>DETECTED SCENE CATEGORIZATION:</h3>
            <h1 style='margin:10px 0; font-size: 3rem; color:#FFFFFF;'>{predicted_class.upper()}</h1>
            <p style='margin:0; font-size:1.1rem;'>📈 Prediction Probability Certainty: <strong>{confidence:.2f}%</strong></p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Visual Progress Bar
    st.progress(int(confidence))