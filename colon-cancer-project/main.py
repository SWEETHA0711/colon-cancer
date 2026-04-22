# =========================
# IMPORTS
# =========================
import streamlit as st
from ultralytics import YOLO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Colon Cancer Detection", layout="centered")

st.title("🧠 Colon Cancer Detection System")
st.write("Upload an image to detect polyps and analyze cancer risk")

# =========================
# HELPER: Draw boxes with PIL (no cv2 needed)
# =========================
def draw_box(img_array, box, label, color_rgb):
    """Draw bounding box and label on image using PIL."""
    img_pil = Image.fromarray(img_array.copy())
    draw = ImageDraw.Draw(img_pil)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)
    # Draw label background
    text_bbox = draw.textbbox((x1, y1 - 20), label)
    draw.rectangle(text_bbox, fill=color_rgb)
    draw.text((x1, y1 - 20), label, fill="white")
    return np.array(img_pil)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("models/best.pt")

    try:
        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
    except Exception:
        efficientnet = models.efficientnet_b0(pretrained=False)
    efficientnet = torch.nn.Sequential(*list(efficientnet.children())[:-1])
    efficientnet.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    with open("models/rf_cancer_model.pkl", "rb") as f:
        rf_model = pickle.load(f)

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

    # Convert image — ensure RGB (drop alpha if PNG has it)
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    detection_output = img_rgb.copy()
    final_output = img_rgb.copy()

    # =========================
    # YOLO DETECTION
    # =========================
    with st.spinner("Running polyp detection..."):
        results = yolo_model.predict(img_rgb, conf=0.25)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    # =========================
    # STEP 1: DETECTION
    # =========================
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        label = f"Polyp {score * 100:.1f}%"
        detection_output = draw_box(detection_output, (x1, y1, x2, y2), label, (255, 0, 0))

    st.subheader("🩺 Step 1: Polyp Detection")
    if len(boxes) == 0:
        st.info("No polyps detected in this image.")
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

        if high_risk > low_risk:
            color_rgb = (255, 0, 0)   # RED
            risk_label = "HIGH RISK"
        else:
            color_rgb = (0, 200, 0)   # GREEN
            risk_label = "LOW RISK"

        label = f"{pred} | {risk_label}"
        final_output = draw_box(final_output, (x1, y1, x2, y2), label, color_rgb)

        st.image(crop, caption=f"Polyp {i + 1}: {pred} — {risk_label}", width=200)

    # =========================
    # STEP 3: RISK GRAPH
    # =========================
    if len(all_high_risk) > 0:

        st.subheader("📊 Risk Analysis")

        fig, ax = plt.subplots(figsize=(8, 4))

        x = np.arange(len(all_high_risk))
        width = 0.35

        ax.bar(x - width / 2, all_high_risk, width, label='High Risk', color='red')
        ax.bar(x + width / 2, all_low_risk, width, label='Low Risk', color='green')

        ax.set_xlabel("Polyp Index")
        ax.set_ylabel("Probability")
        ax.set_title("Cancer Risk Analysis")
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{i + 1}" for i in range(len(x))])
        ax.set_ylim(0, 1)
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

    # =========================
    # FINAL OUTPUT
    # =========================
    st.subheader("🧾 Final Classification Output")
    st.image(final_output, use_container_width=True)
