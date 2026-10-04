import matplotlib.pylab as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# ==========================================
# PHẦN 1: Tải dữ liệu và hiển thị 25 ảnh mẫu
# ==========================================
digits = load_digits()
print("Kích thước dữ liệu gốc:", digits.data.shape)

np.random.seed(1)
fig = plt.figure(figsize=(4, 4))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05
)

j = 1
for i in np.random.choice(digits.data.shape[0], 25):
  plt.subplot(5, 5, j)
  plt.imshow(np.reshape(digits.data[i, :], (8, 8)), cmap='binary')
  plt.axis('off')
  j += 1
plt.suptitle('25 ảnh chữ số mẫu', y=1.05)
plt.show()

# ==========================================
# PHẦN 2: Giảm chiều 2D và vẽ Scatter Plot (Đoạn mã mới)
# ==========================================
# 1. Giảm chiều dữ liệu từ 64D -> 2D
pca_digits = PCA(2)
digits.data_proj = pca_digits.fit_transform(digits.data)

# 2. In tỷ lệ thông tin giữ lại
print(
    "Tỷ lệ biến thiên giữ lại (Explained Variance):",
    np.sum(pca_digits.explained_variance_ratio_),
)

# 3. Vẽ biểu đồ phân tán 2D
plt.figure(figsize=(12, 8))
plt.scatter(
    digits.data_proj[:, 0],
    digits.data_proj[:, 1],
    lw=0.25,
    c=digits.target,
    edgecolor='k',
    s=100,
    cmap=plt.colormaps['cubehelix'].resampled(10),  # Cập nhật chuẩn Colab mới
)
plt.xlabel('PC1', size=14)
plt.ylabel('PC2', size=14)
plt.title('2D Projection of handwritten digits with PCA', size=16)

cbar = plt.colorbar(ticks=range(10), label='digit value')
plt.clim(-0.5, 9.5)
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()