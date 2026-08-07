import os
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

# 1. Cấu hình giao diện ứng dụng Streamlit
st.set_page_config(
    page_title="Retinal AI - Sàng lọc Bệnh lý Đáy mắt & RP",
    page_icon="👁️",
    layout="wide"
)

# Thêm CSS Tùy chỉnh phong cách y tế hiện đại
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .warning-card {
        background-color: #FEF2F2;
        border-left: 5px solid #DC2626;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">👁️ Trợ lý AI Chẩn đoán Bệnh lý Đáy mắt</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hệ thống Hỗ trợ Sàng lọc sớm Bệnh Viêm võng mạc sắc tố (RP) & Bệnh lý Đáy mắt dựa trên EfficientNetB0</div>', unsafe_allow_html=True)

# 2. Hàm tải mô hình AI (Sử dụng Cache để tối ưu hiệu năng)
@st.cache_resource
def load_trained_model():
    # Tìm đường dẫn file model trong thư mục models/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '..', 'models', 'best_efficientnet_finetuned.keras')
    
    if not os.path.exists(model_path):
        model_path = os.path.join('models', 'best_efficientnet_finetuned.keras')
        
    model = tf.keras.models.load_model(model_path)
    return model

try:
    with st.spinner('Đang khởi tạo mô hình AI...'):
        model = load_trained_model()
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}. Vui lòng kiểm tra file 'models/best_efficientnet_finetuned.keras'.")

# 3. Danh sách nhãn lớp bệnh lý (Khớp thứ tự huấn luyện)
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
    'Retinitis Pigmentosa': 'Viêm võng mạc sắc tố - Bệnh di truyền gây suy giảm thị lực ban đêm và thu hẹp thị trường.',
    'Retinal Detachment': 'Bong võng mạc - Tình trạng cấp tính võng mạc tách khỏi lớp mô phía dưới.',
    'Diabetic Retinopathy': 'Bệnh võng mạc đái tháo đường - Biến chứng tổn thương mạch máu do đường huyết cao.',
    'Disc Edema': 'Phù đĩa thị - Tình trạng sưng đĩa thị giác do tăng áp lực nội sọ hoặc viêm nhiễm.',
    'Glaucoma': 'Bệnh Cườm nước - Bệnh lý gây tổn thương dây thần kinh thị giác do tăng nhãn áp.',
    'Central Serous Chorioretinopathy': 'Bệnh hắc võng mạc trung tâm thanh dịch - Tích tụ dịch dưới võng mạc.',
    'Macular Scar': 'Sẹo hoàng điểm - Tổn thương vùng hoàng điểm ảnh hưởng thị lực trung tâm.',
    'Myopia': 'Cận thị tiến triển - Biến đổi hình thái đáy mắt do trục nhãn cầu dài.',
    'Healthy': 'Đáy mắt bình thường - Không phát hiện dấu hiệu bệnh lý bất thường.'
}

# 4. Bố cục Giao diện (2 Cột)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Tải ảnh đáy mắt")
    uploaded_file = st.file_uploader("Chọn tệp ảnh đáy mắt (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đáy mắt đã tải lên", use_container_width=True)

with col2:
    st.subheader("📊 Kết quả Phân tích AI")
    if uploaded_file is not None:
        if st.button("🔍 Tiến hành Chẩn đoán", type="primary", use_container_width=True):
            with st.spinner("Đang phân tích hình ảnh..."):
                # Tiền xử lý ảnh chuẩn đầu vào (224, 224, 3)
                img_rgb = image.convert("RGB")
                img_resized = ImageOps.fit(img_rgb, (224, 224), Image.Resampling.LANCZOS)
                img_array = np.asarray(img_resized, dtype=np.float32)
                img_tensor = np.expand_dims(img_array, axis=0)

                # Dự đoán
                predictions = model.predict(img_tensor, verbose=0)[0]
                top_idx = np.argmax(predictions)
                top_class = CLASS_NAMES[top_idx]
                confidence = predictions[top_idx] * 100

                # Hiển thị kết quả chính
                st.markdown(f"""
                <div class="result-card">
                    <h3 style="color: #15803D; margin: 0;">Kết quả: {top_class}</h3>
                    <p style="font-size: 1.2rem; font-weight: bold; margin-top: 5px;">Độ tin cậy: {confidence:.2f}%</p>
                    <p style="color: #374151;"><em>{CLASS_DESCRIPTIONS.get(top_class, '')}</em></p>
                </div>
                """, unsafe_allow_html=True)

                # Biểu đồ phân bố xác suất các lớp
                st.write("---")
                st.write("**Phân bố Xác suất các Bệnh lý:**")
                prob_df = pd.DataFrame({
                    'Bệnh lý': CLASS_NAMES,
                    'Xác suất (%)': predictions * 100
                }).sort_values(by='Xác suất (%)', ascending=True)

                st.bar_chart(prob_df.set_index('Bệnh lý'))

                # Cảnh báo y tế
                st.markdown("""
                <div class="warning-card">
                    ⚠️ <strong>Lưu ý Y tế:</strong> Kết quả chẩn đoán từ mô hình AI chỉ mang tính chất tham khảo và hỗ trợ sàng lọc ban đầu. Quyết định điều trị cuối cùng phải thuộc về Bác sĩ chuyên khoa Mắt.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Vui lòng chọn và tải lên ảnh đáy mắt ở cột bên trái để thực hiện chẩn đoán.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #9CA3AF;'>Dự án AI Y tế — Đồ án Nghiên cứu Chẩn đoán Bệnh lý Võng mạc & RP (EfficientNetB0)</p>", unsafe_allow_html=True)