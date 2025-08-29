import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import joblib
from model_arch import MobileeNetV2
import numpy as np
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MobileNetV2 model
@st.cache_resource
def load_mobilenet_model():
    model = MobileeNetV2(num_classes=4).to('cpu')
    model.load_state_dict(torch.load(f"models/chicken_es_mobilenet_CNN_moredata.pt", weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load RandomForest model
@st.cache_resource
def load_rf_model():
    model = joblib.load('chicken_disease_classifier.pkl')
    return model

mobilenet_model = load_mobilenet_model()
rf_model = load_rf_model()

# Define the class names
class_names = ['cocci', 'healthy', 'ncd', 'salmo']

# Define preprocessing transformations for MobileNetV2
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to match the input size of your model
    transforms.Lambda(lambda x: x.convert("RGB")),  # Convert image to RGB
    transforms.ToTensor(),          # Convert image to tensor
])

# Streamlit App
st.title("Chicken Disease Detector")
st.write("Upload a chicken excreta sample and choose a model for prediction.")

# Option to choose the model
model_choice = st.selectbox("Select model for prediction:", ["MobileNetV2", "RandomForest"])

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image for both models
    input_image = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
    
    # Random Forest Classifier requires feature extraction, so dummy features are used here
    # Replace this with actual feature extraction for Random Forest
    features = np.array([1, 1, 1, 1, 1])  # Replace with real feature extraction
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.reshape(-1, 1)).flatten()  # Reshaping for scaler
    
    # Predict using the chosen model
    if model_choice == "MobileNetV2":
        # Predict using MobileNetV2 model
        with torch.no_grad():
            output = mobilenet_model(input_image)
            _, predicted = torch.max(output, 1)  # Get the index of the highest score
            predicted_class = class_names[predicted.item()]
    else:
        # Predict using RandomForest model
        prediction = rf_model.predict([features_scaled])
        predicted_class = class_names[prediction[0]]

    # Display the result
    st.write(f"### Prediction: {predicted_class}")
