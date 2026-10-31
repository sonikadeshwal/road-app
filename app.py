import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Road Damage Detector")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
f = st.file_uploader("Upload a road photo", type=["jpg", "jpeg", "png"])

if f:
    img = Image.open(f).convert("RGB")
    r = model.predict(img, conf=0.25)[0]
    st.image(r.plot()[:, :, ::-1], caption=f"{len(r.boxes)} damage(s) found")