import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# 1. Tải dữ liệu khuôn mặt mẫu (ví dụ Olivetti faces 64x64)
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.data  # Kích thước (400, 4096)

# 2. Định nghĩa Pipeline và đặt tên bước là 'pca'
pca = PCA(n_components=150, whiten=True, random_state=42)
pipeline = Pipeline([('pca', pca)])

# 3. Huấn luyện (fit) pipeline trên dữ liệu
pipeline.fit(X)

# 4. Trích xuất danh sách các thành phần chính (Eigenvectors) từ bước PCA
eigenfaces = pipeline.named_steps['pca'].components_

# 5. Hiển thị 10 Eigenfaces đầu tiên
fig = plt.figure(figsize=(10, 4.5))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.05)

for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
    # Reshape vector 4096 chiều thành ma trận ảnh 64x64
    ax.imshow(
        np.reshape(eigenfaces[i, :], (64, 64)),
        cmap='bone',
        interpolation='nearest',
    )
    ax.set_title(f'Eigenface {i+1}', fontsize=10)

plt.show()