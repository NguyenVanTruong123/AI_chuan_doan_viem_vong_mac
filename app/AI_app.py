import os
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import cv2

# 1. Cấu hình trang Streamlit
st.set_page_config(
    page_title="Retinal AI - Chẩn đoán Bệnh lý Võng mạc & RP",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS Tùy chỉnh Phong cách Y tế Chuyên nghiệp & Glassmorphism UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #1E3A8A 0%, #0284C7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #4B5563;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 1.8rem;
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .rp-badge {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #F59E0B;
    }
    
    .result-card-success {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border-left: 6px solid #16A34A;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .result-card-warning {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border-left: 6px solid #DC2626;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
        transform: translateY(-1px);
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

# 4. Hàm phát sinh Grad-CAM Heatmap
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

# 5. Danh mục Nhãn & Mô tả Y tế
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
    'Retinitis Pigmentosa': '🎯 **Viêm võng mạc sắc tố (RP)**: Bệnh lý di truyền thoái hóa tế bào thụ cảm ánh sáng. AI đạt **Recall 96%** trong phát hiện bệnh này.',
    'Retinal Detachment': '🚨 **Bong võng mạc**: Tình trạng cấp tính neurosensory retina tách khỏi lớp biểu mô sắc tố. Cần can thiệp phẫu thuật khẩn cấp.',
    'Disc Edema': '👁️ **Phù đĩa thị**: Sưng phù đĩa thị giác do tăng áp lực nội sọ hoặc viêm dây thần kinh thị giác.',
    'Diabetic Retinopathy': '🩸 **Võng mạc đái tháo đường**: Tổn thương hệ mạch máu nhỏ võng mạc do biến chứng tiểu đường.',
    'Glaucoma': '👁️ **Glaucoma (Cườm nước)**: Bệnh lý gây tổn thương dây thần kinh thị giác liên quan đến tăng nhãn áp.',
    'Central Serous Chorioretinopathy': '💧 **Hắc võng mạc trung tâm thanh dịch**: Tích tụ dịch subretinal dưới hoàng điểm.',
    'Macular Scar': '🔍 **Sẹo hoàng điểm**: Vùng tổn thương xơ hóa tại hoàng điểm gây suy giảm thị lực trung tâm.',
    'Myopia': '👓 **Cận thị tiến triển**: Biến đổi thoái hóa màng trạch và võng mạc do trục nhãn cầu quá dài.',
    'Healthy': '✅ **Mắt khỏe mạnh**: Đáy mắt bình thường, chưa phát hiện dấu hiệu tổn thương bệnh lý.'
}

# 6. Sidebar (Thanh bên thông tin & Thống kê)
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/ophthalmology.png", width=70)
    st.markdown("### 👁️ Retinal AI Platform")
    st.markdown("---")
    st.markdown("**Thông số Mô hình:**")
    st.markdown("- **Backbone:** EfficientNetB0 (Transfer Learning)")
    st.markdown("- **Độ chính xác toàn cục (Test Acc):** `78.3%`")
    st.markdown("- **Recall bệnh RP:** `<font color='#16A34A'><b>96.0%</b></font>`", unsafe_allow_html=True)
    st.markdown("- **Macro F1-Score:** `81.0%`")
    st.markdown("- **Dữ liệu sạch:** `10,948` ảnh (Đã khử rò rỉ MD5)")
    st.markdown("---")
    st.info("💡 **Ghi chú:** Mô hình sử dụng kỹ thuật Explainable AI (Grad-CAM) để trực quan hóa vùng đĩa thị & hoàng điểm mà AI tập trung phân tích.")

# 7. Header Chính
st.markdown('<div class="main-title">👁️ Hệ thống AI Chẩn đoán Bệnh lý Đáy mắt</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sàng lọc sớm Viêm võng mạc sắc tố (RP) & Bệnh lý Võng mạc với Trực quan hóa Grad-CAM (XAI)</div>', unsafe_allow_html=True)

# 8. Bố cục Tabs Chức năng
tab1, tab2, tab3 = st.tabs(["🏥 Phân tích Ảnh Đáy Mắt", "📚 Thư viện Bệnh lý Võng mạc", "📈 Thông số Đánh giá Mô hình"])

# TAB 1: PHÂN TÍCH CHẨN ĐOÁN
with tab1:
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.subheader("📤 Upload Ảnh Đáy mắt")
        uploaded_file = st.file_uploader("Chọn ảnh đáy mắt (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh đáy mắt tải lên", use_container_width=True)

    with col2:
        st.subheader("📊 Kết quả Phân tích & Trực quan hóa")
        if uploaded_file is not None:
            if st.button("🔍 Tiến hành Chẩn đoán & Tạo Grad-CAM", type="primary", use_container_width=True):
                with st.spinner("Đang phân tích hình ảnh & phát sinh bản đồ nhiệt Grad-CAM..."):
                    # Tiền xử lý ảnh
                    img_rgb = image.convert("RGB")
                    img_resized = ImageOps.fit(img_rgb, (224, 224), Image.Resampling.LANCZOS)
                    img_array = np.asarray(img_resized, dtype=np.float32)
                    img_tensor = np.expand_dims(img_array, axis=0)

                    # Dự đoán
                    predictions = model.predict(img_tensor, verbose=0)[0]
                    top_idx = np.argmax(predictions)
                    top_class = CLASS_NAMES[top_idx]
                    confidence = predictions[top_idx] * 100

                    # Hiển thị thẻ kết quả
                    card_style = "result-card-warning" if top_class == "Retinitis Pigmentosa" else "result-card-success"
                    st.markdown(f"""
                    <div class="{card_style}">
                        <h3 style="margin: 0; color: #1E293B;">Chẩn đoán: {top_class}</h3>
                        <h4 style="margin-top: 5px; color: #2563EB;">Độ tin cậy: {confidence:.2f}%</h4>
                        <p style="margin-top: 8px;">{CLASS_DESCRIPTIONS.get(top_class, '')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Tạo Grad-CAM Heatmap
                    heatmap = generate_gradcam(img_tensor, model)
                    if heatmap is not None:
                        st.write("---")
                        st.write("**🔥 Trực quan hóa Vùng chú ý AI (Grad-CAM Overlay):**")
                        
                        orig_np = np.uint8(img_array)
                        resized_heatmap = cv2.resize(heatmap, (224, 224))
                        resized_heatmap = np.uint8(255 * resized_heatmap)
                        color_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
                        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
                        superimposed = cv2.addWeighted(orig_np, 0.6, color_heatmap, 0.4, 0)

                        cam_col1, cam_col2 = st.columns(2)
                        with cam_col1:
                            st.image(color_heatmap, caption="Grad-CAM Heatmap", use_container_width=True)
                        with cam_col2:
                            st.image(superimposed, caption="Vùng tập trung chẩn đoán", use_container_width=True)

                    # Phân bố xác suất
                    st.write("---")
                    st.write("**Phân bố Xác suất 9 Lớp Bệnh lý:**")
                    prob_df = pd.DataFrame({
                        'Bệnh lý': CLASS_NAMES,
                        'Xác suất (%)': predictions * 100
                    }).sort_values(by='Xác suất (%)', ascending=True)

                    st.bar_chart(prob_df.set_index('Bệnh lý'))

        else:
            st.info("👈 Vui lòng tải lên ảnh đáy mắt ở cột bên trái để thực hiện chẩn đoán.")

# TAB 2: THƯ VIỆN BỆNH LÝ
with tab2:
    st.subheader("📚 Thống kê & Triệu chứng Các Lớp Bệnh lý Võng mạc")
    for cls_name, desc in CLASS_DESCRIPTIONS.items():
        with st.expander(f"📌 {cls_name}"):
            st.write(desc)

# TAB 3: THÔNG SỐ NGHIÊN CỨU
with tab3:
    st.subheader("📈 Hiệu năng Thực nghiệm trên Tập Test Độc lập (1,622 Ảnh)")
    st.markdown("""
    | Bệnh lý | Precision | **Recall (Độ nhạy)** | **F1-Score** |
    | :--- | :---: | :---: | :---: |
    | 🎯 **Viêm võng mạc sắc tố (RP)** | **0.85** | **0.96 (96%)** | **0.90 (90%)** |
    | 👁️ **Bong võng mạc** | **0.97** | **0.97 (97%)** | **0.97 (97%)** |
    | 👁️ **Phù đĩa thị** | **0.91** | **0.97 (97%)** | **0.94 (94%)** |
    | 🩸 **Võng mạc đái tháo đường** | **0.95** | **0.88** | **0.91 (91%)** |
    | 👁️ **Hắc võng mạc thanh dịch (CSC)** | 0.82 | 0.69 | 0.75 |
    | 👁️ **Cận thị tiến triển** | 0.79 | 0.68 | 0.73 |
    | 👁️ **Sẹo hoàng điểm** | 0.67 | 0.75 | 0.71 |
    | 👁️ **Mắt khỏe mạnh** | 0.63 | 0.85 | 0.72 |
    | 👁️ **Glaucoma** | 0.71 | 0.55 | 0.62 |
    | **TRUNG BÌNH TỔNG THỂ (Accuracy)** | — | — | **78.3%** |
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6B7280;'>© 2026 Medical AI Research Project — EfficientNetB0 & Grad-CAM Retinal Pathology Screening Platform</p>", unsafe_allow_html=True)