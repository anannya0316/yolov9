import streamlit as st
from PIL import Image
import numpy as np

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.transforms import functional as F

# Load pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    return model

model = load_model()
model.eval()

# Function to make prediction
def make_prediction(img, model):
    img = F.to_tensor(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img)
    return prediction[0]

# Function to display prediction
def display_prediction(img, prediction):
    boxes = prediction["boxes"].cpu().numpy().astype(int)
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Adjust confidence threshold as needed
            box = box.astype(np.int32)
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            draw.text((box[0], box[1]), f"{label} ({score:.2f})", fill="red")
            del draw
    st.image(img, caption="Object Detection", use_column_width=True)

# Main function
def main():
    st.title("Object Detection")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Detecting objects...")

        prediction = make_prediction(img, model)
        display_prediction(np.array(img), prediction)

if __name__ == "__main__":
    main()
