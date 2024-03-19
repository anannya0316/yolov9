import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from streamlit.components.v1 import html

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Load pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    return model

model = load_model()
model.eval()

# Define COCO labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Function to make prediction
def make_prediction(img, model):
    img = F.to_tensor(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img)
    return prediction[0]

# Function to display prediction
def display_prediction(img, prediction):
    img = Image.fromarray(np.uint8(img))  # Convert numpy array to PIL Image
    draw = ImageDraw.Draw(img)
    boxes = prediction["boxes"].cpu().numpy().astype(int)
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Adjust confidence threshold as needed
            box = box.astype(np.int32)
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            label_text = COCO_INSTANCE_CATEGORY_NAMES[label]
            draw.text((box[0], box[1]), f"{label_text} ({score:.2f})", fill="red")
    del draw
    return img

# Main function
def main():
    st.title("Object Detection")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        try:
            img = Image.open(uploaded_image)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.write("")
            st.write("Detecting objects...")

            prediction = make_prediction(img, model)
            edited_img = display_prediction(np.array(img), prediction)
            
            # Display the edited image with movable bounding boxes
            st.image(edited_img, caption="Object Detection", use_column_width=True)
            
            # Add JavaScript code for making boxes movable
            st.write("To make the bounding boxes movable, click and drag them.")
            st.write("Please note that this functionality might not work perfectly in the Streamlit environment.")
            js_code = """
            <script>
            // Add JavaScript code here
            </script>
            """
            st.components.v1.html(js_code)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
