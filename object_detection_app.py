import streamlit as st
from PIL import Image, ImageDraw
import numpy as np

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline="red", width=2)
    return image

# Streamlit app
def main():
    st.title("Editable Bounding Boxes")

    # Upload image
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Image", use_column_width=True)

        # Define initial bounding boxes
        boxes = [(100, 100, 200, 200)]  # Example initial bounding box coordinates (x_min, y_min, x_max, y_max)

        # Display editable bounding boxes
        edited_image = draw_bounding_boxes(image.copy(), boxes)
        st.image(edited_image, caption="Editable Bounding Boxes", use_column_width=True)

        # Interactive widget to modify bounding box coordinates
        x_min, y_min, x_max, y_max = st.slider("Bounding Box Coordinates", 0, image.width, (100, 200, 200, 300))
        boxes[0] = (x_min, y_min, x_max, y_max)

        # Button to save edited image
        if st.button("Save Edited Image"):
            edited_image_with_boxes = draw_bounding_boxes(image.copy(), boxes)
            edited_image_with_boxes.save("edited_image.jpg", "JPEG")  # Save edited image to downloads folder
            st.success("Edited image saved successfully.")

if __name__ == "__main__":
    main()
