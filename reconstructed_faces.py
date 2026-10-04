import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# ==========================================
# 1. TẢI DỮ LIỆU & TÍNH MEAN / SD (Z-SCORE)
# ==========================================
# Tải tập dữ liệu 400 khuôn mặt (mỗi ảnh 64x64 = 4096 pixels)
faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)
faces = faces_data.data  # Kích thước: (400, 4096)

# Tính giá trị trung bình (mean) và độ lệch chuẩn (SD) cho từng pixel
mean_vector = np.mean(faces, axis=0)
sd_vector = np.std(faces, axis=0)

# Reshape thành dạng ảnh 64x64 để chuẩn bị khôi phục
mean_face = np.reshape(mean_vector, (64, 64))
sd_face = np.reshape(sd_vector, (64, 64))

# Chuẩn hóa Z-score cho dữ liệu gốc
faces_scaled = (faces - mean_vector) / sd_vector


# ==========================================
# 2. CHẠY PCA VÀ GIẢM CHIỀU DỮ LIỆU (64D)
# ==========================================
pca = PCA(n_components=64, random_state=42)
pipeline = Pipeline([('pca', pca)])

# Chiếu dữ liệu từ 4096 chiều xuống 64 chiều (faces_proj)
faces_proj = pipeline.fit_transform(faces_scaled)


# ==========================================
# 3. TÁI TẠO ẢNH TỪ 64 EIGENFACES (CODE CỦA BẠN)
# ==========================================

# 1. Đảo ngược phép biến đổi PCA: Đưa dữ liệu 64D về lại không gian 4096D
faces_inv_proj = pipeline.named_steps['pca'].inverse_transform(faces_proj)

# 2. Reshape vector 4096 chiều thành ma trận dạng ảnh (400 ảnh, 64x64 pixels)
faces_reconstructed = np.reshape(faces_inv_proj, (400, 64, 64))

# 3. Hiển thị 25 ảnh đã được tái tạo
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

np.random.seed(0)  # Giữ nguyên random seed=0 để so sánh với 25 ảnh gốc
j = 1
for i in np.random.choice(range(faces.shape[0]), 25):
    ax = fig.add_subplot(5, 5, j, xticks=[], yticks=[])

    # Đảo ngược bước Z-score normalization: Nhân lại SD và cộng lại Mean Face
    img_restored = mean_face + sd_face * faces_reconstructed[i, :]

    ax.imshow(img_restored, cmap='bone', interpolation='nearest')
    j += 1

plt.suptitle(
    '25 khuôn mặt tái tạo (Reconstructed) từ 64 Eigenfaces',
    y=1.03,
    fontsize=12,
    fontweight='bold',
)
plt.show()