import gzip
import os
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 1. HÀM TẢI DỮ LIỆU TỪ MÁY CHỦ S3
# ==========================================
def download(filename, source='https://ossci-datasets.s3.amazonaws.com/mnist/'):
    if not os.path.exists(filename):
        print(f'Đang tải {filename}...')
        urlretrieve(source + filename, filename)
        print(f'Đã tải xong {filename}!')


# ==========================================
# 2. HÀM ĐỌC MA TRẬN ẢNH VÀ NHÃN TỪ FILE GZ
# ==========================================
def load_mnist_images(filename):
    download(filename)
    with gzip.open(filename, 'rb') as f:
        # Bỏ qua 16 bytes header của file binary MNIST images
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 784)  # Reshape thành dạng vector 784 (28x28)


def load_mnist_labels(filename):
    download(filename)
    with gzip.open(filename, 'rb') as f:
        # Bỏ qua 8 bytes header của file binary MNIST labels
        return np.frombuffer(f.read(), np.uint8, offset=8)


# ==========================================
# 3. TẢI TẬP KIỂM THỬ (10,000 ẢNH TEST & NHÃN)
# ==========================================
print("--- Đang chuẩn bị tập dữ liệu MNIST Test ---")
test_data = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

print(f"Tổng số ảnh test: {test_data.shape[0]}")
print(f"Kích thước mỗi ảnh: {test_data.shape[1]} pixels (28x28)")


# ==========================================
# 4. HIỂN THỊ 25 ẢNH ĐẦU TIÊN
# ==========================================
plt.figure(figsize=(9, 9))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(test_data[i].reshape(28, 28), cmap='gray')
    plt.title(f'True Label: {test_labels[i]}', fontsize=10, fontweight='bold')
    plt.axis('off')

plt.suptitle(
    '25 ảnh chữ số viết tay đầu tiên từ tập Test kèm nhãn thực tế (Ground-Truth)',
    fontsize=12,
    fontweight='bold',
)
plt.tight_layout()
plt.show()