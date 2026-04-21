# =========================
# IMPORTS
# =========================
import streamlit as st
from ultralytics import YOLO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pickle
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Colon Cancer Detection", layout="centered")

st.title("🧠 Colon Cancer Detection System")
st.write("Upload an image to detect polyps and analyze cancer risk")

# =========================
# LOAD MODELS (FIXED PATHS)
# =========================
@st.cache_resource
def load_models():

    BASE_DIR = os.path.dirname(__file__)

    # 🔥 FIXED PATH (Cloud + Local safe)
    yolo_path = os.path.join(BASE_DIR, "colon-cancer-project", "models", "best.pt")
    rf_path = os.path.join(BASE_DIR, "colon-cancer-project", "models", "rf_cancer_model.pkl")

    yolo_model = YOLO(yolo_path)

    efficientnet = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )
    efficientnet = torch.nn.Sequential(*list(efficientnet.children())[:-1])
    efficientnet.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    rf_model = pickle.load(open(rf_path, "rb"))

    return yolo_model, efficientnet, transform, rf_model


yolo_model, efficientnet, transform, rf_model = load_models()

# =========================
# UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader(
    "📤 Drag & Drop or Click to Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img_rgb = np.array(image)

    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    detection_output = img_rgb.copy()
    final_output = img_rgb.copy()

    # =========================
    # YOLO DETECTION
    # =========================
    results = yolo_model.predict(img_rgb, conf=0.25)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    # =========================
    # STEP 1: DETECTION
    # =========================
    for box, score in zip(boxes, scores):

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(detection_output, (x1,y1), (x2,y2), (255,0,0), 2)

        label = f"Polyp {score*100:.1f}%"
        cv2.putText(detection_output, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    st.subheader("🩺 Step 1: Polyp Detection")
    st.image(detection_output, use_container_width=True)

    # =========================
    # STEP 2: CLASSIFICATION
    # =========================
    st.subheader("🔍 Step 2: Polyp Classification")

    all_high_risk = []
    all_low_risk = []

    for i, (box, score) in enumerate(zip(boxes, scores)):

        x1, y1, x2, y2 = map(int, box)
        crop = img_rgb[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        input_tensor = transform(crop).unsqueeze(0)

        with torch.no_grad():
            feat = efficientnet(input_tensor).squeeze().numpy().reshape(1, -1)

        pred = rf_model.predict(feat)[0]
        probs = rf_model.predict_proba(feat)[0]

        class_names = rf_model.classes_
        prob_dict = dict(zip(class_names, probs))

        high_risk = prob_dict.get("Adenomatous", 0) + prob_dict.get("Serrated_Lesions", 0)
        low_risk = prob_dict.get("Hyperplastic", 0)

        all_high_risk.append(high_risk)
        all_low_risk.append(low_risk)

        # =========================
        # COLOR FIXED
        # =========================
        if high_risk > low_risk:
            color = (0, 0, 255)   # 🔴 RED (BGR)
            risk_label = "HIGH RISK"
        else:
            color = (0, 255, 0)   # 🟢 GREEN
            risk_label = "LOW RISK"

        label = f"{pred} | {risk_label}"

        cv2.rectangle(final_output, (x1,y1), (x2,y2), color, 2)
        cv2.putText(final_output, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        st.image(crop, caption=f"Polyp {i+1}: {pred}", width=200)

    # =========================
    # STEP 3: RISK GRAPH
    # =========================
    if len(all_high_risk) > 0:

        st.subheader("📊 Risk Analysis")

        fig, ax = plt.subplots()

        x = np.arange(len(all_high_risk))
        width = 0.35

        ax.bar(x - width/2, all_high_risk, width, label='High Risk', color='red')
        ax.bar(x + width/2, all_low_risk, width, label='Low Risk', color='green')

        ax.set_xlabel("Polyp Index")
        ax.set_ylabel("Probability")
        ax.set_title("Cancer Risk Analysis")
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{i+1}" for i in range(len(x))])
        ax.set_ylim(0,1)
        ax.legend()

        st.pyplot(fig)

    # =========================
    # FINAL OUTPUT
    # =========================
    st.subheader("🧾 Final Classification Output")
    st.image(final_output, use_container_width=True)
