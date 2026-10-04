import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Tải tập dữ liệu Olivetti Faces
faces = fetch_olivetti_faces().data

# 2. Xây dựng Pipeline: Chuẩn hóa Z-score + PCA 64 thành phần
n_comp = 64
pipeline = Pipeline(
    [('scaling', StandardScaler()), ('pca', PCA(n_components=n_comp))]
)

# 3. Thực thi Pipeline và giảm chiều dữ liệu từ 4096D -> 64D
faces_proj = pipeline.fit_transform(faces)
print(f"Kích thước dữ liệu sau khi nén: {faces_proj.shape}")  # (400, 64)

# 4. Trích xuất Mean Face và SD Face từ StandardScaler
mean_face = np.reshape(pipeline.named_steps['scaling'].mean_, (64, 64))
sd_face = np.reshape(np.sqrt(pipeline.named_steps['scaling'].var_), (64, 64))

# 5. Biểu đồ 1: Đồ thị tỷ lệ biến thiên tích lũy (Cumulative Explained Variance)
cum_variance = np.cumsum(pipeline.named_steps['pca'].explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(cum_variance, linewidth=2, color='navy')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(0, n_comp)
plt.xlabel('Số thành phần chính (n_components)', fontsize=11)
plt.ylabel('Tỷ lệ biến thiên tích lũy (Cumulative Variance)', fontsize=11)
plt.title(
    'Lượng thông tin giữ lại theo số thành phần chính',
    fontsize=12,
    fontweight='bold',
)
plt.show()

print(f"Tổng lượng thông tin giữ lại với 64 PCs: {cum_variance[-1]*100:.2f}%")

# 6. Biểu đồ 2: Hiển thị Mean Face và SD Face
plt.figure(figsize=(10, 4.5))

plt.subplot(1, 2, 1)
plt.imshow(mean_face, cmap='bone')
plt.axis('off')
plt.title('Mean Face (Khuôn mặt trung bình)', fontsize=11, fontweight='bold')

plt.subplot(1, 2, 2)
plt.imshow(sd_face, cmap='bone')
plt.axis('off')
plt.title('SD Face (Độ lệch chuẩn)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()