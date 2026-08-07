# 👁️ AI Retinal Disease Classification & Retinitis Pigmentosa (RP) Screening

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

An end-to-end Deep Learning framework for multi-class retinal fundus image classification, specifically optimized for the early detection and screening of **Retinitis Pigmentosa (RP)** and other critical ocular pathologies.

---

## 📌 Project Overview

**Retinitis Pigmentosa (RP)** is a rare, genetic degenerative eye disease causing progressive vision loss. Early diagnosis via retinal fundus examination is crucial to preserve patient sight. 

This repository provides a complete, production-ready Machine Learning pipeline featuring:
- **Cleaned Data Engineering**: Strict MD5 hash verification removing 345 data-leakage duplicate images across Train/Valid/Test splits.
- **Two-Stage Transfer Learning**: Fine-tuning an **EfficientNetB0** architecture with adaptive Class Weights to overcome severe class imbalance.
- **Explainable AI (XAI)**: Grad-CAM heatmap visualization identifying critical anatomical regions (optic disc, macular area, pigmentary deposits).
- **Interactive Web UI**: Real-time inference application built with Streamlit.

---

## 📊 Dataset & Class Distribution

The dataset is derived from the **Eye-Fundus-10** benchmark. To ensure focus on retinal fundus pathologies, non-fundus surface diseases (*Pterygium*) were excluded, resulting in **9 target classes**:

| Disease Class | Clean Train Count | Clinical Significance |
| :--- | :---: | :--- |
| **Retinitis Pigmentosa (RP)** | 283 | Target genetic retinal dystrophy |
| **Diabetic Retinopathy** | 1,194 | Microvascular diabetes complication |
| **Glaucoma** | 970 | Optic nerve atrophy & elevated IOP |
| **Healthy** | 891 | Normal fundus baseline |
| **Myopia** | 756 | Progressive axial elongation |
| **Macular Scar** | 652 | Central visual acuity impairment |
| **Disc Edema** | 267 | Optic disc swelling / intracranial pressure |
| **Retinal Detachment** | 261 | Acute neurosensory retina detachment |
| **Central Serous Chorioretinopathy** | 201 | Subretinal fluid accumulation |
| **Total Clean Images** | **10,948** | *Filtered via MD5 checksums* |

---

## 🛠️ Model Architecture & Training Protocol

```
Input Image (224x224x3)
       │
       ▼
Data Augmentation (Flip, Rotation, Zoom, Contrast, Brightness)
       │
       ▼
EfficientNetB0 Backbone (Pretrained ImageNet Weights)
       │
       ▼
Global Average Pooling 2D
       │
       ▼
BatchNormalization -> Dropout (0.4) -> Dense (256, ReLU) -> BatchNormalization -> Dropout (0.3)
       │
       ▼
Dense (9 classes, Softmax)
```

### Two-Stage Training Protocol:
1. **Stage 1 (Frozen Backbone)**: Train Top Classifier layers for 15 epochs with Adam (`lr = 1e-3`).
2. **Stage 2 (Fine-Tuning)**: Unfreeze the last 30 layers of EfficientNetB0 and train for 15 epochs with Adam (`lr = 1e-4`).
3. **Imbalance Mitigation**: Compute balanced Class Weights applied directly to `CategoricalCrossentropy` loss.

---

## 🏆 Benchmark Results (Independent Test Set: 1,622 Images)

The model was evaluated on a strictly isolated test set **free of data leakage**:

- **Overall Test Accuracy**: **78.3%**
- **Macro Average F1-Score**: **0.81 (81.0%)**

### Class-level Performance Metrics:

| Disease Class | Precision | **Recall (Sensitivity)** | **F1-Score** |
| :--- | :---: | :---: | :---: |
| 🎯 **Retinitis Pigmentosa (RP)** | **0.85** | **0.96 (96.0%)** | **0.90 (90.0%)** |
| 👁️ **Retinal Detachment** | **0.97** | **0.97 (97.0%)** | **0.97 (97.0%)** |
| 👁️ **Disc Edema** | **0.91** | **0.97 (97.0%)** | **0.94 (94.0%)** |
| 🩸 **Diabetic Retinopathy** | **0.95** | **0.88** | **0.91 (91.0%)** |
| 👁️ **Central Serous Chorioretinopathy** | 0.82 | 0.69 | 0.75 |
| 👁️ **Myopia** | 0.79 | 0.68 | 0.73 |
| 👁️ **Macular Scar** | 0.67 | 0.75 | 0.71 |
| 👁️ **Healthy** | 0.63 | 0.85 | 0.72 |
| 👁️ **Glaucoma** | 0.71 | 0.55 | 0.62 |

> 🌟 **Key Clinical Insight**: The model achieves an exceptional **96% Recall (Sensitivity)** for **Retinitis Pigmentosa**, ensuring virtually zero false-negatives in clinical screening.

---

## 📁 Repository Structure

```
.
├── app/
│   └── AI_app.py                          # Streamlit Web Application
├── notebooks/
│   └── btl-sang-loc-benh-vong-mac.ipynb   # Main Jupyter Notebook (Data cleaning, Model, Grad-CAM)
├── models/
│   └── best_efficientnet_finetuned.keras  # Saved Model Checkpoint
├── docs/
│   └── Phan_loai_benh_Vong_Mac.docx       # Project Research Report
├── .gitignore                             # Excluded Dataset & Checkpoint Rules
├── LICENSE                                # MIT Open-Source License
├── README.md                              # Project Documentation
└── requirements.txt                       # Python Dependencies
```

---

## 🚀 Quickstart & Web Application

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/retinal-disease-classification-rp.git
cd retinal-disease-classification-rp
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit Web Application
```bash
streamlit run app/AI_app.py
```
Open your browser at `http://localhost:8501` to test the AI prediction interface!

---

## 🔬 Explainable AI (Grad-CAM)

To provide clinical transparency, **Grad-CAM (Gradient-weighted Class Activation Mapping)** extracts feature maps from the final convolutional layer (`top_activation`), overlaying heatmaps to highlight visual cues used by the model:

```python
# Grad-CAM heatmap generation code snippet in Jupyter Notebook
heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer_name="top_activation")
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
