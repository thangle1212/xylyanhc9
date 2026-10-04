import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle

from skimage.io import imread

from time import time


# ==========================================
# 1. ĐỌC ẢNH TỪ MÁY TÍNH
# ==========================================

# Thay 'pepper.jpg' bằng đường dẫn đến file ảnh trên máy của bạn
filename = 'pepper.jpg' 

# Đọc ảnh
pepper = imread(filename)

print("Ảnh:", filename)
print("Kích thước:", pepper.shape)


# ==========================================
# 2. HIỂN THỊ ẢNH GỐC
# ==========================================

plt.figure(figsize=(8, 6))
plt.axis('off')

plt.title(
    'Original image (%d colors)' %
    len(np.unique(pepper))
)

plt.imshow(pepper)
plt.show()


# ==========================================
# 3. CHUYỂN ẢNH SANG FLOAT
# ==========================================

pepper = np.array(pepper, dtype=np.float64) / 255


# ==========================================
# 4. CHUYỂN ẢNH THÀNH MẢNG 2D
# ==========================================

w, h, d = original_shape = tuple(pepper.shape)

assert d == 3

image_array = np.reshape(
    pepper,
    (w * h, d)
)


# ==========================================
# 5. HÀM TẠO LẠI ẢNH
# ==========================================

def recreate_image(codebook, labels, w, h):
    """Recreate the compressed image"""

    d = codebook.shape[1]

    image = np.zeros((w, h, d))

    label_idx = 0

    for i in range(w):
        for j in range(h):

            image[i][j] = codebook[
                labels[label_idx]
            ]

            label_idx += 1

    return image


# ==========================================
# 6. K-MEANS
# ==========================================

plt.figure(figsize=(10, 10))
plt.clf()

i = 1

for k in [64, 32, 16, 4]:

    print("\n==============================")
    print("K =", k)
    print("==============================")

    t0 = time()

    plt.subplot(2, 2, i)
    plt.axis('off')

    # Lấy 1000 pixel ngẫu nhiên để train K-Means
    image_array_sample = shuffle(
        image_array,
        random_state=0
    )[:1000]

    kmeans = KMeans(
        n_clusters=k,
        random_state=0
    ).fit(image_array_sample)

    print(
        "K-Means done in %0.3fs."
        % (time() - t0)
    )


    # Dự đoán màu cho toàn bộ pixel
    print(
        "Predicting color indices on the full image..."
    )

    t0 = time()

    labels = kmeans.predict(image_array)

    print(
        "Prediction done in %0.3fs."
        % (time() - t0)
    )


    # Hiển thị ảnh sau khi giảm màu
    plt.title(
        'Quantized image (%d colors, K-Means)' % k
    )

    plt.imshow(
        recreate_image(
            kmeans.cluster_centers_,
            labels,
            w,
            h
        )
    )

    i += 1

plt.show()


# ==========================================
# 7. RANDOM COLOR QUANTIZATION
# ==========================================

plt.figure(figsize=(10, 10))
plt.clf()

i = 1

for k in [64, 32, 16, 4]:

    print("\n==============================")
    print("Random K =", k)
    print("==============================")


    plt.subplot(2, 2, i)
    plt.axis('off')


    # Chọn k màu ngẫu nhiên
    codebook_random = shuffle(
        image_array,
        random_state=0
    )[:k + 1]


    print(
        "Predicting color indices on the full image (random)"
    )

    t0 = time()


    labels_random = pairwise_distances_argmin(
        codebook_random,
        image_array,
        axis=0
    )


    print(
        "done in %0.3fs."
        % (time() - t0)
    )


    plt.title(
        'Quantized image (%d colors, Random)' % k
    )


    plt.imshow(
        recreate_image(
            codebook_random,
            labels_random,
            w,
            h
        )
    )


    i += 1

plt.show()