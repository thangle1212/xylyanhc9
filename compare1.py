# ============================================
# IMAGE SEGMENTATION
# K-MEANS + SPECTRAL CLUSTERING
# ============================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize





# ==========================================
# 1. ĐỌC ẢNH TỪ MÁY TÍNH
# ==========================================

# Thay 'pepper.jpg' bằng đường dẫn đến file ảnh trên máy của bạn
filename = 'keanu-reeves.jpg' 

# Đọc ảnh
im = imread(filename)


# Nếu ảnh có alpha channel (RGBA) thì bỏ alpha
if im.ndim == 3 and im.shape[-1] == 4:
    im = im[:, :, :3]

# Nếu ảnh grayscale thì chuyển thành RGB
if im.ndim == 2:
    im = np.stack((im,) * 3, axis=-1)

# Resize ảnh về 100 x 100
im = resize(
    im,
    (100, 100, 3),
    anti_aliasing=True
)

print("Ảnh:", filename)
print("Kích thước:", im.shape)


# ============================================
# 2. CONVERT TO GRAYSCALE
# ============================================

img = rgb2gray(im)


# ============================================
# 3. PREPARE DATA
# ============================================

k = 2

# Chuyển ảnh thành danh sách pixel
X = np.reshape(
    im,
    (-1, im.shape[-1])
)

print("Số pixel:", X.shape[0])
print("Số đặc trưng:", X.shape[1])


# ============================================
# 4. K-MEANS SEGMENTATION
# ============================================

two_means = cluster.MiniBatchKMeans(
    n_clusters=k,
    random_state=10
)

two_means.fit(X)

y_pred = two_means.predict(X)

labels = np.reshape(
    y_pred,
    im.shape[:2]
)


# ============================================
# 5. SPECTRAL CLUSTERING
# ============================================

spectral = cluster.SpectralClustering(
    n_clusters=k,
    eigen_solver='arpack',
    affinity='nearest_neighbors',
    n_neighbors=100,
    random_state=10
)

spectral.fit(X)

# np.int đã bị bỏ → dùng int
y_pred = spectral.labels_.astype(int)

labels_spectral = np.reshape(
    y_pred,
    im.shape[:2]
)


# ============================================
# 6. DISPLAY RESULTS
# ============================================

plt.figure(figsize=(20, 20))


# --------------------------------------------
# K-Means segmentation
# --------------------------------------------

plt.subplot(2, 2, 1)

plt.imshow(labels, cmap='gray')

plt.title(
    'K-Means Segmentation (k=2)',
    size=20
)

plt.axis('off')


# --------------------------------------------
# K-Means contour
# --------------------------------------------

plt.subplot(2, 2, 2)

plt.imshow(im)

plt.contour(
    labels == 0,
    levels=[0.5],
    colors='red'
)

plt.title(
    'K-Means Contour (k=2)',
    size=20
)

plt.axis('off')


# --------------------------------------------
# Spectral segmentation
# --------------------------------------------

plt.subplot(2, 2, 3)

plt.imshow(
    labels_spectral,
    cmap='gray'
)

plt.title(
    'Spectral Segmentation (k=2)',
    size=20
)

plt.axis('off')


# --------------------------------------------
# Spectral contour
# --------------------------------------------

plt.subplot(2, 2, 4)

plt.imshow(im)

plt.contour(
    labels_spectral == 0,
    levels=[0.5],
    colors='red'
)

plt.title(
    'Spectral Contour (k=2)',
    size=20
)

plt.axis('off')


plt.tight_layout()
plt.show()