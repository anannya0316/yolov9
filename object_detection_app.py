import streamlit as st
from PIL import Image
import numpy as np

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

# Load default weights and categories
DEFAULT_WEIGHTS = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
DEFAULT_CATEGORIES = DEFAULT_WEIGHTS.meta["categories"]

# Function to load model with optional weights and categories
@st.cache
def load_model(weights=DEFAULT_WEIGHTS, categories=DEFAULT_CATEGORIES):
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()  # Set model for evaluation
    return model, categories

# Function to make prediction given an image and model
def make_prediction(img, model):
    img_preprocess = model.transform_input(img)  # Preprocess image
    prediction = model(img_preprocess)[0]  # Perform prediction
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

# Function to create image with bounding boxes and labels
def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img.transpose(2, 0, 1))  # Transpose image tensor
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"],
                                          labels=prediction["labels"], width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np

# Function to display detection results
def display_results(image, prediction):
    st.image(image, use_column_width=True)
    st.header("Detection Results")
    if prediction["labels"]:
        st.write(prediction["labels"])
    else:
        st.write("No objects detected.")

# Dashboard
st.title("Object Detector")

# Model selection
model_weights = st.sidebar.selectbox("Select Model Weights", [DEFAULT_WEIGHTS, "Custom"])
if model_weights == "Custom":
    # Allow user to upload custom weights file
    custom_weights_file = st.sidebar.file_uploader("Upload Custom Weights File", type=["pt"])
    if custom_weights_file is not None:
        model_weights = custom_weights_file

# Load model
if model_weights:
    model, categories = load_model(model_weights)
    st.sidebar.write("Model loaded successfully.")

    # Threshold adjustment
    threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.05)

    # Image upload
    upload = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if upload is not None:
        # Process uploaded image
        img = Image.open(upload)

        # Make prediction
        prediction = make_prediction(np.array(img), model)
        filtered_prediction = {k: v for k, v in prediction.items() if k != "boxes" and v}

        # Display detection results
        display_results(img, filtered_prediction)
