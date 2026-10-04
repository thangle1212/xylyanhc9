import gzip
import os
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 1. HÀM TẢI VÀ NẠP DỮ LIỆU MNIST
# ==========================================
def download(filename, source='https://ossci-datasets.s3.amazonaws.com/mnist/'):
    if not os.path.exists(filename):
        print(f'Đang tải {filename}...')
        urlretrieve(source + filename, filename)


def load_mnist_images(filename):
    download(filename)
    with gzip.open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)


def load_mnist_labels(filename):
    download(filename)
    with gzip.open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


# Nạp dữ liệu Train (60,000 ảnh)
print('--- ĐANG NẠP DỮ LIỆU TRAIN MNIST ---')
train_data = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')


# ==========================================
# 2. ĐỊNH NGHĨA HÀM HIỂN THỊ
# ==========================================
def display_char(image, label_title=''):
    plt.figure(figsize=(3, 3))
    plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
    if label_title:
        plt.title(label_title, fontsize=11, fontweight='bold')
    plt.axis('off')
    plt.show()


# ==========================================
# 3. HUẤN LUYỆN GENERATIVE MODEL (GAUSSIAN BAYES)
# ==========================================
def fit_generative_model(x, y):
    k = 10  # Số nhãn lớp (từ 0 đến 9)
    d = x.shape[1]  # Số lượng đặc trưng (784 pixel)
    mu = np.zeros((k, d))
    sigma = np.zeros((k, d, d))
    pi = np.zeros(k)
    c = 3500  # Hằng số chuẩn hóa ma trận hiệp phương sai (Smoothing factor)

    for label in range(k):
        indices = y == label
        pi[label] = sum(indices) / float(len(y))
        mu[label] = np.mean(x[indices, :], axis=0)
        sigma[label] = np.cov(x[indices, :], rowvar=0, bias=1) + c * np.eye(d)

    return mu, sigma, pi


# 4. Huấn luyện mô hình
print('Đang tính toán các thông số mu, sigma, pi cho Generative Model...')
mu, sigma, pi = fit_generative_model(train_data, train_labels)
print('Hoàn thành!\n')

# 5. Hiển thị ảnh trung bình (Mean Images) cho các chữ số 0, 1, 2
display_char(mu[0], 'Mean Image: Digit 0')
display_char(mu[1], 'Mean Image: Digit 1')
display_char(mu[2], 'Mean Image: Digit 2')