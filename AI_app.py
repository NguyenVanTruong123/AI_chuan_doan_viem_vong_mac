import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. Cấu hình trang
st.set_page_config(page_title="Chẩn đoán Bệnh Mắt AI", page_icon="👁️")

st.title("👁️ Trợ lý AI Chẩn đoán Bệnh Mắt")
st.write("Hệ thống sàng lọc sớm sử dụng mô hình EfficientNetB0")

# 2. Hàm tải model (Cache để không phải load lại mỗi lần f5)
@st.cache_resource
def load_model():
    # Thay 'eye_disease_model.h5' bằng tên file model thực tế của bạn
    model = tf.keras.models.load_model('final_model_run.keras')
    return model

with st.spinner('Đang tải mô hình AI...'):
    model = load_model()

# 3. Định nghĩa nhãn (Labels) - Cần khớp thứ tự với lúc train (One-hot encoding) 
class_names = [  
    'Central Serous Chorioretinopathy',    
    'Diabetic Retinopathy',                          
    'Disc Edema',                          
    'Glaucoma',                            
    'Healthy',                                                                             
    'Macular Scar',
    'Myopia',
    'Pterygium',
    'Retinal Detachment',
    'Retinitis Pigmentosa'                 
]
# LƯU Ý: Bạn hãy sửa lại danh sách này đúng thứ tự thư mục lúc train nhé!

# 4. Giao diện tải ảnh
uploaded_file = st.file_uploader("Chọn ảnh đáy mắt (JPG, PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Hiển thị ảnh
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
    
    # Nút dự đoán
    if st.button('🔍 Phân tích ngay'):
        with st.spinner('Đang xử lý...'):
            # --- TIỀN XỬ LÝ ẢNH (QUAN TRỌNG) ---
            # 1. Convert sang RGB (đề phòng ảnh xám hoặc PNG 4 kênh)
            image = image.convert("RGB")
            
            # 2. Resize về 224x224 (Như trong slide của bạn)
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            
            # 3. Chuyển sang mảng numpy
            img_array = np.asarray(image)
            
            # 4. Chuẩn hóa (Nếu lúc train bạn dùng rescale 1./255 thì bỏ comment dòng dưới)
            # img_array = img_array / 255.0
            
            # 5. Mở rộng chiều (Batch dimension) -> (1, 224, 224, 3)
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = img_array

            # --- DỰ ĐOÁN ---
            prediction = model.predict(data)
            index = np.argmax(prediction) # Lấy vị trí có xác suất cao nhất
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # --- HIỂN THỊ KẾT QUẢ ---
            st.success(f"Kết quả: **{class_name}**")
            st.info(f"Độ tin cậy: **{confidence_score * 100:.2f}%**")
            
            # Hiển thị biểu đồ xác suất (Optional)
            st.bar_chart(prediction[0])