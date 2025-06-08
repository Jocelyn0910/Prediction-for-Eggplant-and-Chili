import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import json

# Page configuration
st.set_page_config(page_title="Pest and Disease Prediction for Eggplant and Chili", page_icon="🌿", layout="centered")

st.title("🌿 Pest and Disease Prediction for Eggplant and Chili")
st.write("Upload an image to detect diseases in **eggplant** or **chili** leaves.")

# Load metadata
with open("Eggplant_Cause_and_Remedy.json", "r") as f:
    Eggplant_Cause_and_Remedy = json.load(f)
with open("Chili_Cause_and_Remedy.json", "r") as f:
    Chili_Cause_and_Remedy = json.load(f)

# User selects plant type
plant_type = st.selectbox("Select the crop to detect:", ["Eggplant", "Chili"])

# Load the appropriate model and metadata
if plant_type == "Eggplant":
    model_path = "best_eggplant_model_yolo11s_v5.pt"
    metadata = Eggplant_Cause_and_Remedy
    model_path = "chili_detection_3.pt"
    metadata = Chili_Cause_and_Remedy

model = YOLO(model_path)

# Image upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img_copy = img.copy()

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    # Inference
    st.subheader("🔍 Prediction Result")
    results = model.predict(img, conf=0.5)

    boxes = results[0].boxes
    labels = results[0].names

    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls)
            class_name = labels[cls_id]

            info = metadata.get(class_name, {
                "label": class_name,
                "cause": "No cause info available.",
                "remedy": "No remedy info available."
            })

            label_display = info["label"]
            cause = info["cause"]
            remedy = info["remedy"]

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()

            # Draw box and label
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_copy, f"{label_display} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display info in Streamlit
            st.success(f"🦠 **Prediction:** {label_display}")
            st.write(f"🧪 **Cause:** {cause}")
            st.write(f"💡 **Remedy:** {remedy}")

        st.image(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)
    else:
        st.warning("No disease or pest detected. Try another image.")