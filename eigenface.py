import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# 1. Tải tập dữ liệu Olivetti Faces
dataset = fetch_olivetti_faces()
faces = dataset.data

print(f"Kích thước tập dữ liệu: {faces.shape}")
# Output: (400, 4096) -> 400 ảnh, mỗi ảnh chứa 64x64 = 4096 điểm ảnh

# 2. Hiển thị 25 khuôn mặt ngẫu nhiên từ tập dữ liệu
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

np.random.seed(0)
j = 1
for i in np.random.choice(range(faces.shape[0]), 25):
    ax = fig.add_subplot(5, 5, j, xticks=[], yticks=[])
    # Chuyển vector 4096 phần tử thành ma trận ảnh 64x64
    ax.imshow(np.reshape(faces[i, :], (64, 64)), cmap=plt.cm.bone, interpolation='nearest')
    j += 1

plt.suptitle("25 khuôn mặt mẫu từ Olivetti Dataset", y=1.03, fontsize=12, fontweight='bold')
plt.show()