from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

st.set_page_config(page_title="Road Damage Detector", layout="wide")

MODEL_PATH = Path(__file__).parent / "best.pt"


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def ransac_plane(P, thr=0.01, iters=300, seed=0):
    rng = np.random.default_rng(seed)
    best_count, best = -1, None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ a
        count = int((np.abs(P @ n + d) < thr).sum())
        if count > best_count:
            best_count, best = count, (n, d)
    return best


def severity(depth_mm):
    if depth_mm < 25:
        return "LOW"
    if depth_mm <= 50:
        return "MEDIUM"
    return "HIGH"


def run_lidar_sim(depth1_mm, depth2_mm):
    rng = np.random.default_rng(0)
    N = 60000
    x, y = rng.uniform(0, 10, N), rng.uniform(0, 4, N)
    z = rng.normal(0, 0.002, N)
    for cx, cy, r, d in [(3, 1.5, 0.25, depth1_mm / 1000), (7, 2.8, 0.18, depth2_mm / 1000)]:
        z[(x - cx) ** 2 + (y - cy) ** 2 < r ** 2] -= d
    P = np.c_[x, y, z]
    n, off = ransac_plane(P)
    dist = np.abs(P @ n + off)
    mask = dist > 0.015
    labels = DBSCAN(eps=0.08, min_samples=8).fit_predict(P[mask][:, :2])
    area_per_point = (10 * 4) / N
    rows = []
    for k in sorted(set(labels) - {-1}):
        m = labels == k
        depth_mm = float(dist[mask][m].max() * 1000)
        rows.append({"pothole": k + 1, "depth_mm": round(depth_mm),
                     "area_cm2": round(float(m.sum() * area_per_point * 10000)),
                     "severity": severity(depth_mm)})
    return pd.DataFrame(rows), P, dist


st.title("Road Surface Crack and Distress Detection")
tab1, tab2 = st.tabs(["Image detection", "LiDAR depth demo (simulated)"])

with tab1:
    if not MODEL_PATH.exists():
        st.error("best.pt was not found next to app.py. Upload it to the same folder "
                 "(exact name: best.pt) and reboot the app.")
        st.stop()

    try:
        model = load_model()
    except Exception as e:
        st.error("Model load nahi hua. Error neeche hai:")
        st.exception(e)
        st.stop()
    names = model.names  # class names come from the trained model itself
    st.write(f"Upload a road photo. A YOLOv8s model detects: **{', '.join(names.values())}**.")

    conf = st.slider("Confidence threshold (lower finds more, with more false alarms)",
                     0.05, 0.9, 0.2, 0.05)
    f = st.file_uploader("Road photo", type=["jpg", "jpeg", "png"])

    if f:
        img = Image.open(f).convert("RGB")
        r = model.predict(img, conf=conf, imgsz=640, verbose=False)[0]
        st.image(r.plot()[:, :, ::-1], caption=f"{len(r.boxes)} damage(s) found")
        if len(r.boxes):
            df = pd.DataFrame({"damage": [names[int(c)] for c in r.boxes.cls],
                               "confidence": r.boxes.conf.cpu().numpy()})
            st.dataframe(df.groupby("damage").agg(count=("damage", "size"),
                                                  mean_confidence=("confidence", "mean")).round(2))
        else:
            st.info("No damage found. Try lowering the confidence threshold.")

with tab2:
    st.write("A simulated 10 m by 4 m road with two potholes. The method fits the road plane (RANSAC), "
             "groups the damaged points (DBSCAN) and rates each pothole by depth. "
             "This is simulated data, not a real scan.")
    d1 = st.slider("Pothole 1 true depth (mm)", 20, 100, 60)
    d2 = st.slider("Pothole 2 true depth (mm)", 20, 100, 30)
    if st.button("Run LiDAR simulation"):
        table, P, dist = run_lidar_sim(d1, d2)
        st.dataframe(table, hide_index=True)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        sc = ax.scatter(P[:, 0], P[:, 1], c=dist, s=0.4, cmap="viridis", rasterized=True)
        fig.colorbar(sc, ax=ax, label="distance from road plane (m)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        st.pyplot(fig)
