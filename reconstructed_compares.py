import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# ==========================================
# BƯỚC 1: TẢI DỮ LIỆU VÀ CHUẨN HÓA (Z-SCORE)
# ==========================================
print("Đang tải dữ liệu khuôn mặt Olivetti...")
faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)
faces = faces_data.data  # Mảng (400, 4096)

# Tính Mean và SD theo từng pixel
mean_vector = np.mean(faces, axis=0)
sd_vector = np.std(faces, axis=0)

mean_face = np.reshape(mean_vector, (64, 64))
sd_face = np.reshape(sd_vector, (64, 64))

# Chuẩn hóa Z-score
faces_scaled = (faces - mean_vector) / sd_vector


# ==========================================
# BƯỚC 2: TẠO PIPELINE & GIẢM CHIỀU VỚI PCA (64D)
# ==========================================
print("Đang huấn luyện PCA với 64 thành phần chính...")
pca = PCA(n_components=64, random_state=42)
pipeline = Pipeline([('pca', pca)])

# Chiếu dữ liệu xuống không gian 64 chiều (faces_proj)
faces_proj = pipeline.fit_transform(faces_scaled)


# ==========================================
# BƯỚC 3: TÁI TẠO VÀ SO SÁNH ẢNH GỐC VS TÁI TẠO
# ==========================================
# 1. Trích xuất ảnh gốc thứ nhất (4096 chiều -> 64x64)
orig_face = np.reshape(faces[0, :], (64, 64))

# 2. Tái tạo lại ảnh bằng phép nhân ma trận (@)
reconst_face = np.reshape(
    faces_proj[0, :] @ pipeline.named_steps['pca'].components_, (64, 64)
)

# 3. Đảo ngược chuẩn hóa Z-score
reconst_face = mean_face + sd_face * reconst_face

# 4. Hiển thị so sánh trực quan sóng đôi (Side-by-side)
plt.figure(figsize=(10, 5))

# Ảnh gốc
plt.subplot(1, 2, 1)
plt.imshow(orig_face, cmap='bone', interpolation='nearest')
plt.axis('off')
plt.title('Original (Ảnh gốc)', size=16, fontweight='bold')

# Ảnh tái tạo
plt.subplot(1, 2, 2)
plt.imshow(reconst_face, cmap='bone', interpolation='nearest')
plt.axis('off')
plt.title('Reconstructed (Ảnh tái tạo)', size=16, fontweight='bold')

plt.tight_layout()
plt.show()