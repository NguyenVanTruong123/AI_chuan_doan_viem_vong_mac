import os
import io
import base64
import random
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.cm as cm

# 1. Cấu hình trang Streamlit
st.set_page_config(
    page_title="RetinalAI - Hệ thống AI Chẩn đoán Đáy mắt & RP",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Siêu CSS Custom Design System (Futuristic Cyber-Medical Dark Theme & Glassmorphism)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e293b 0%, #0f172a 50%, #020617 100%);
        color: #f8fafc;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .hud-banner {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 242, 254, 0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .hud-title {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
    }
    
    .hud-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 10px #34d399;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 12px 24px -10px rgba(56, 189, 248, 0.2);
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }

    .diagnosis-box {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 1.8rem;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    }
    
    .rp-warning {
        border-left: 6px solid #f43f5e;
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.15) 0%, rgba(15, 23, 42, 0.8) 100%);
    }
    
    .normal-success {
        border-left: 6px solid #10b981;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.8) 100%);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre;
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        color: #94a3b8;
        font-weight: 600;
        padding: 0 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0284c7 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
    }

    div[data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.3);
        border: 2px dashed rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        padding: 1rem;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.35);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%);
        box-shadow: 0 8px 25px rgba(2, 132, 199, 0.5);
        transform: translateY(-2px);
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0b0f17;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Tải mô hình AI với Cache
@st.cache_resource
def load_trained_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '..', 'models', 'best_efficientnet_finetuned.keras')
    if not os.path.exists(model_path):
        model_path = os.path.join('models', 'best_efficientnet_finetuned.keras')
    return tf.keras.models.load_model(model_path)

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"⚠️ Lỗi khởi tạo mô hình: {e}. Vui lòng kiểm tra file 'models/best_efficientnet_finetuned.keras'.")
    st.stop()

# 4. Hàm lấy danh sách tất cả ảnh trong bộ dữ liệu (Local eye/ hoặc Sample)
@st.cache_data
def get_all_dataset_images():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_dir, '..')
    
    image_paths = []
    local_eye = os.path.join(project_root, 'eye')
    if os.path.exists(local_eye):
        for root, dirs, files in os.walk(local_eye):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, f))
    
    if not image_paths:
        sample_dir = os.path.join(base_dir, 'sample_images')
        if os.path.exists(sample_dir):
            for f in os.listdir(sample_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(sample_dir, f))
                    
    return image_paths

# 5. Hàm phát sinh Grad-CAM Heatmap
def generate_gradcam(img_tensor, model):
    try:
        base_model = model.get_layer('efficientnetb0')
        last_conv_layer = base_model.get_layer('top_activation')
        grad_model = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=[last_conv_layer.output, base_model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_tensor)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = last_conv_layer_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception:
        return None

def overlay_heatmap_pil(heatmap, orig_image):
    jet = cm.get_cmap("jet")
    jet_colors = jet(heatmap)[:, :, :3]
    jet_heatmap = Image.fromarray(np.uint8(255 * jet_colors))
    
    resized_heatmap = jet_heatmap.resize((224, 224), Image.Resampling.LANCZOS)
    orig_resized = orig_image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    
    superimposed = Image.blend(orig_resized, resized_heatmap, alpha=0.45)
    return resized_heatmap, superimposed

# 6. Danh mục Nhãn & Mô tả Y tế
CLASS_NAMES = [
    'Central Serous Chorioretinopathy',
    'Diabetic Retinopathy',
    'Disc Edema',
    'Glaucoma',
    'Healthy',
    'Macular Scar',
    'Myopia',
    'Retinal Detachment',
    'Retinitis Pigmentosa'
]

CLASS_DESCRIPTIONS = {
    'Retinitis Pigmentosa': '🎯 **Viêm võng mạc sắc tố (RP)**: Bệnh lý di truyền gây suy giảm thị lực ban đêm và thu hẹp thị trường. Mô hình đạt **Recall 96.0%** tuyệt đối trong sàng lọc sớm.',
    'Retinal Detachment': '🚨 **Bong võng mạc**: Tình trạng cấp tính võng mạc tách khỏi lớp mô biểu mô. Cần can thiệp phẫu thuật khẩn cấp.',
    'Disc Edema': '👁️ **Phù đĩa thị**: Tình trạng sưng phù đĩa thị giác do tăng áp lực nội sọ hoặc viêm dây thần kinh.',
    'Diabetic Retinopathy': '🩸 **Võng mạc đái tháo đường**: Tổn thương hệ vi mạch võng mạc do biến chứng tiểu đường.',
    'Glaucoma': '👁️ **Glaucoma (Cườm nước)**: Bệnh lý tổn thương dây thần kinh thị giác liên quan đến tăng nhãn áp.',
    'Central Serous Chorioretinopathy': '💧 **Hắc võng mạc trung tâm thanh dịch**: Tích tụ dịch subretinal dưới hoàng điểm.',
    'Macular Scar': '🔍 **Sẹo hoàng điểm**: Vùng tổn thương xơ hóa tại hoàng điểm ảnh hưởng thị lực trung tâm.',
    'Myopia': '👓 **Cận thị tiến triển**: Biến đổi thoái hóa màng trạch và võng mạc do trục nhãn cầu dài.',
    'Healthy': '✅ **Mắt khỏe mạnh**: Đáy mắt bình thường, không phát hiện dấu hiệu bất thường.'
}

# 7. Sidebar Navigation & Info
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem;">👁️</div>
            <h2 style="color: #38bdf8; margin-top: 0.5rem; font-weight: 800;">RetinalAI</h2>
            <p style="color: #64748b; font-size: 0.85rem;">Clinical Decision Support Platform</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### ⚙️ Engine Specs")
    st.markdown("""
    - **Architecture:** `EfficientNetB0`
    - **Resolution:** `224 x 224 px`
    - **Clean Dataset:** `10,948` samples
    - **Target Classes:** `9 Pathologies`
    """)
    st.markdown("---")
    
    st.markdown("### 🏆 Core Metrics")
    st.markdown("""
    - **RP Recall:** <span style="color:#34d399; font-weight:bold;">96.0%</span>
    - **Overall Acc:** <span style="color:#38bdf8; font-weight:bold;">78.3%</span>
    - **Macro F1:** <span style="color:#818cf8; font-weight:bold;">81.0%</span>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("© 2026 Medical AI Research Group. Powered by TensorFlow & Grad-CAM.")

# 8. Header HUD Banner
st.markdown("""
    <div class="hud-banner">
        <div>
            <div class="hud-title">RetinalAI Diagnostic Suite</div>
            <div class="hud-subtitle">Hệ thống AI Sàng lọc Bệnh lý Võng mạc & Viêm võng mạc sắc tố (RP) với Grad-CAM Explainable AI</div>
        </div>
        <div class="status-badge">
            <div class="pulse-dot"></div>
            SYSTEM ONLINE
        </div>
    </div>
""", unsafe_allow_html=True)

# 9. Top Glassmorphism Metrics Bar
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown("""
        <div class="glass-card">
            <div class="metric-lbl">RP Sensitivity</div>
            <div class="metric-val" style="color: #34d399;">96.0%</div>
        </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown("""
        <div class="glass-card">
            <div class="metric-lbl">Macro F1-Score</div>
            <div class="metric-val" style="color: #38bdf8;">81.0%</div>
        </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown("""
        <div class="glass-card">
            <div class="metric-lbl">Clean Test Set</div>
            <div class="metric-val" style="color: #818cf8;">1,622</div>
        </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown("""
        <div class="glass-card">
            <div class="metric-lbl">XAI Interpretability</div>
            <div class="metric-val" style="color: #c084fc;">Grad-CAM</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 10. Main Tabs
tab1, tab2, tab3 = st.tabs(["🔬 Diagnostic Terminal", "📚 Clinical Knowledgebase", "📊 Model Performance"])

# TAB 1: DIAGNOSTIC TERMINAL
with tab1:
    c1, c2 = st.columns([1, 1], gap="large")

    all_dataset_images = get_all_dataset_images()

    if 'selected_sample' not in st.session_state:
        st.session_state.selected_sample = None

    def pick_random_sample_callback():
        if all_dataset_images:
            st.session_state.selected_sample = random.choice(all_dataset_images)

    with c1:
        st.markdown("### 📤 Upload Fundus Scan")
        uploaded_file = st.file_uploader("Drag and drop your retinal fundus image (JPG, PNG)...", type=["jpg", "jpeg", "png"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Nút lấy ảnh ngẫu nhiên có callback on_click để đảm bảo luôn bốc ảnh mới mỗi lần click!
        st.button("🎲 CHỌN ẢNH NGẦU NHIÊN TỪ BỘ DỮ LIỆU", on_click=pick_random_sample_callback)

        # Xử lý nguồn ảnh được chọn
        target_image = None
        if uploaded_file is not None:
            target_image = Image.open(uploaded_file)
            st.session_state.selected_sample = None
        elif st.session_state.selected_sample is not None and os.path.exists(st.session_state.selected_sample):
            target_image = Image.open(st.session_state.selected_sample)
            
            parent_folder = os.path.basename(os.path.dirname(st.session_state.selected_sample))
            file_name = os.path.basename(st.session_state.selected_sample)
            st.info(f"🎲 Ảnh ngẫu nhiên từ tập dữ liệu: `{parent_folder}/{file_name}`")

        if target_image is not None:
            st.image(target_image, caption="Current Active Retinal Scan", use_container_width=True)

    with c2:
        st.markdown("### ⚡ AI Inference & Heatmap Analysis")
        if target_image is not None:
            if st.button("🚀 RUN AI DIAGNOSIS & GENERATE GRAD-CAM", key="run_diag"):
                with st.spinner("Processing image tensor & calculating gradient heatmaps..."):
                    img_rgb = target_image.convert("RGB")
                    img_resized = ImageOps.fit(img_rgb, (224, 224), Image.Resampling.LANCZOS)
                    img_array = np.asarray(img_resized, dtype=np.float32)
                    img_tensor = np.expand_dims(img_array, axis=0)

                    # Model prediction
                    predictions = model.predict(img_tensor, verbose=0)[0]
                    top_idx = np.argmax(predictions)
                    top_class = CLASS_NAMES[top_idx]
                    confidence = predictions[top_idx] * 100

                    # Diagnosis Box
                    is_rp = (top_class == "Retinitis Pigmentosa")
                    box_class = "rp-warning" if is_rp else "normal-success"
                    
                    st.markdown(f"""
                        <div class="diagnosis-box {box_class}">
                            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Primary Diagnostic Output</div>
                            <h2 style="color: #f8fafc; margin: 0.3rem 0; font-weight: 800;">{top_class}</h2>
                            <div style="font-size: 1.3rem; font-weight: 700; color: #38bdf8;">Confidence: {confidence:.2f}%</div>
                            <p style="color: #cbd5e1; margin-top: 0.8rem; font-size: 0.95rem;">{CLASS_DESCRIPTIONS.get(top_class, '')}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Grad-CAM Display
                    heatmap = generate_gradcam(img_tensor, model)
                    if heatmap is not None:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("#### 🔥 Grad-CAM Attention Map (XAI Explainability)")
                        
                        resized_heatmap, superimposed = overlay_heatmap_pil(heatmap, target_image)

                        cam_c1, cam_c2 = st.columns(2)
                        with cam_c1:
                            st.image(resized_heatmap, caption="Heatmap (Jet Colormap)", use_container_width=True)
                        with cam_c2:
                            st.image(superimposed, caption="Superimposed Clinical Overlay", use_container_width=True)

                    # Probability Breakdown
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### 📊 Probability Distribution (9 Classes)")
                    
                    prob_df = pd.DataFrame({
                        'Pathology': CLASS_NAMES,
                        'Probability (%)': predictions * 100
                    }).sort_values(by='Probability (%)', ascending=True)

                    st.bar_chart(prob_df.set_index('Pathology'))

        else:
            st.info("👈 Upload a retinal scan or click '🎲 CHỌN ẢNH NGẪU NHIÊN TỪ BỘ DỮ LIỆU' above to test.")

# TAB 2: KNOWLEDGEBASE
with tab2:
    st.markdown("### 📚 Retinal Pathology Reference Catalog")
    for cls_name, desc in CLASS_DESCRIPTIONS.items():
        with st.expander(f"📌 {cls_name}"):
            st.markdown(desc)

# TAB 3: PERFORMANCE METRICS
with tab3:
    st.markdown("### 📈 Comprehensive Evaluation Report (Independent Test Set: 1,622 Images)")
    st.markdown("""
    | Pathology Class | Precision | **Recall (Sensitivity)** | **F1-Score** | Support |
    | :--- | :---: | :---: | :---: | :---: |
    | 🎯 **Retinitis Pigmentosa (RP)** | **0.85** | **0.96 (96.0%)** | **0.90 (90.0%)** | 85 |
    | 👁️ **Retinal Detachment** | **0.97** | **0.97 (97.0%)** | **0.97 (97.0%)** | 75 |
    | 👁️ **Disc Edema** | **0.91** | **0.97 (97.0%)** | **0.94 (94.0%)** | 77 |
    | 🩸 **Diabetic Retinopathy** | **0.95** | **0.88** | **0.91 (91.0%)** | 346 |
    | 👁️ **Central Serous Chorioretinopathy** | 0.82 | 0.69 | 0.75 | 61 |
    | 👁️ **Myopia** | 0.79 | 0.68 | 0.73 | 226 |
    | 👁️ **Macular Scar** | 0.67 | 0.75 | 0.71 | 195 |
    | 👁️ **Healthy** | 0.63 | 0.85 | 0.72 | 268 |
    | 👁️ **Glaucoma** | 0.71 | 0.55 | 0.62 | 289 |
    | **OVERALL ACCURACY** | — | — | **78.3%** | **1,622** |
    """)