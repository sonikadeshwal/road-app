# Automated Road Surface Crack and Distress Detection

We built a system that finds cracks and potholes in road photos and rates how serious a pothole is from LiDAR depth.

- **Image part:** a YOLOv8s model detects and names road damage.
- **LiDAR part:** RANSAC finds the road surface, DBSCAN separates potholes, and the depth sets the severity (LOW, MEDIUM, HIGH).

**Live demo:** https://YOUR-USERNAME-road-damage-detector.hf.space

> The LiDAR part runs on a **simulated** road, because we could not find public road LiDAR with damage labels. See Limitations.

## How it works

1. A road photo goes into YOLOv8s, which returns boxes and damage classes.
2. A road point cloud goes through RANSAC (road plane), a 1.5 cm distance rule (damage points) and DBSCAN (one cluster per pothole).
3. Depth is the largest distance from the road plane. Severity: LOW below 25 mm, MEDIUM 25 to 50 mm, HIGH above 50 mm.

The severity limits are our own choice for illustration. They are not from a road standard.

## Damage classes (RDD2022 subset model)

| Id | Damage type |
|---|---|
| 0 | Longitudinal crack |
| 1 | Transverse crack |
| 2 | Alligator crack |
| 3 | Pothole |
| 4 | Other |

The names of classes 3 and 4 are less certain than the first three. We named them from example images.

## Results

Measured on validation images the models did not train on.

| Experiment | Training images | Epochs | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| A: pothole only (1 class) | 994 | 30 | 0.659 | 0.592 | 0.642 | 0.320 |
| B: RDD2022 subset (5 classes) | 3,000 | 15 | 0.480 | 0.412 | 0.384 | 0.184 |

The two experiments use different data, classes and epochs, so they should not be compared directly.

**LiDAR (simulated road, two potholes):**

| Pothole | Depth, true / found (mm) | Area, true / found (cm²) | Severity |
|---|---|---|---|
| 1 | 60 / 66 | 1,963 / 1,893 | HIGH |
| 2 | 30 / 36 | 1,018 / 1,080 | MEDIUM |

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app: image detection and LiDAR demo |
| `best.pt` | Trained YOLOv8s weights (5 classes) |
| `requirements.txt`, `packages.txt` | Python and system packages for deployment |
| `road_distress_detection.ipynb` | Full notebook: data, training, evaluation, LiDAR |

## Run it yourself

**Streamlit app (on your computer):**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Train again (Kaggle, GPU on, RDD2022 attached):** open `road_distress_detection.ipynb` and click Run All. It takes about 15 minutes on a T4 GPU.

## Limitations

- The LiDAR results come from a simulated scene with a flat road and round potholes. They show the method works, not how it performs on a real sensor.
- The image detections and the LiDAR measurements are not joined yet.
- Models were trained on subsets for a short time, one run each.
- The validation images were also used to choose the best model. The official test split was not used.
- Crack width and length are not measured, because the detector gives boxes, not masks.

## Future work

- Test on real mobile mapping LiDAR.
- Project LiDAR points into the image using calibration (as in KITTI) so each detected box gets a measured depth.
- Train longer on more of RDD2022 and test on the official test split.
- Use segmentation to measure crack width and length.

## Data and references

- RDD2022 road damage dataset (Arya et al., Geoscience Data Journal, 2024).
- Public pothole dataset, 1,243 images: https://github.com/jaygala24/pothole-detection
- YOLOv8, Ultralytics: https://github.com/ultralytics/ultralytics
- Fischler and Bolles, RANSAC, 1981. Ester et al., DBSCAN, 1996.

## Authors

Sonika and Ayush Dobriyal, B.Tech Computer Science and Engineering (AI & ML), Lovely Professional University.
