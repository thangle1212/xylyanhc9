import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. TẢI DỮ LIỆU & HUẤN LUYỆN PCA PIPELINE
# ==========================================
print("Đang tải tập dữ liệu Olivetti faces...")
faces = fetch_olivetti_faces().data  # Kích thước: (400, 4096)

n_comp = 64
pipeline = Pipeline(
    [('scaling', StandardScaler()), ('pca', PCA(n_components=n_comp))]
)

print("Đang chuẩn hóa và chạy PCA...")
faces_proj = pipeline.fit_transform(faces)


# ==========================================
# 2. TRÍCH XUẤT EIGENFACES VÀ TRỌNG SỐ (WEIGHTS)
# ==========================================
pca_step = pipeline.named_steps['pca']
eigenfaces = pca_step.components_  # Kích thước: (64, 4096)

# Chọn ảnh mẫu đầu tiên (index = 0)
sample_idx = 0
original_img = faces[sample_idx].reshape(64, 64)
weights = faces_proj[sample_idx]  # Vector 64 trọng số tương ứng


# ==========================================
# 3. TRỰC QUAN HÓA TỔ HỢP TUYẾN TÍNH
# ==========================================
num_eigenfaces = 4
fig, axes = plt.subplots(1, num_eigenfaces + 1, figsize=(15, 3.2))

# Hiển thị ảnh gốc
axes[0].imshow(original_img, cmap='bone')
axes[0].set_title('original', fontsize=11, fontweight='bold')
axes[0].axis('off')

# Hiển thị Top 4 Eigenfaces và chèn hệ số trọng số ở giữa các hình
for i in range(num_eigenfaces):
    eigenface_img = eigenfaces[i].reshape(64, 64)
    ax = axes[i + 1]
    ax.imshow(eigenface_img, cmap='bone')
    ax.set_title(f'eigenface {i+1}', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Hiển thị ký hiệu toán học và trọng số tương ứng
    prefix = '=  ' if i == 0 else ''
    text_str = f'{prefix}{weights[i]:+.4f}'
    ax.text(
        -0.12,
        0.5,
        text_str,
        transform=ax.transAxes,
        fontsize=10,
        fontweight='bold',
        va='center',
        ha='right',
    )

# Thêm dấu cộng và chấm ba chấm thể hiện tổ hợp các thành phần còn lại
axes[-1].text(
    1.12,
    0.5,
    '+  ...',
    transform=axes[-1].transAxes,
    fontsize=12,
    fontweight='bold',
    va='center',
    ha='left',
)

plt.suptitle(
    'Khai triển ảnh khuôn mặt thành tổ hợp tuyến tính của các Eigenfaces',
    fontsize=13,
    fontweight='bold',
    y=1.05,
)
plt.tight_layout()
plt.show()