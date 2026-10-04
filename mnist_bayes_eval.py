import gzip
import os
import time
from urllib.request import urlretrieve
import numpy as np
from scipy.stats import multivariate_normal


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


# Nạp dữ liệu Train & Test
print('--- ĐANG NẠP DỮ LIỆU TRAIN & TEST MNIST ---')
train_data = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')
test_data = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')


# ==========================================
# 2. HUẤN LUYỆN MÔ HÌNH BAYES (TÍNH MU, SIGMA, PI)
# ==========================================
print('\nĐang huấn luyện mô hình (tính Mean Vector và Covariance Matrix)...')
t0 = time.time()

k = 10  # 10 chữ số từ 0 đến 9
d = train_data.shape[1]  # 784 pixels
mu = np.zeros((k, d))
sigma = np.zeros((k, d, d))
pi = np.zeros(k)
c = 3500  # Smoothing factor cho ma trận hiệp phương sai

for label in range(k):
    indices = train_labels == label
    pi[label] = np.sum(indices) / float(len(train_labels))
    mu[label] = np.mean(train_data[indices, :], axis=0)
    sigma[label] = np.cov(train_data[indices, :], rowvar=0, bias=1) + c * np.eye(d)

print(f'-> Đã hoàn thành huấn luyện trong: {time.time() - t0:.2f} giây')


# ==========================================
# 3. ĐÁNH GIÁ MÔ HÌNH TRÊN 10,000 ẢNH TEST
# ==========================================
print('\nĐang dự đoán nhãn cho 10,000 ảnh test bằng Log-Likelihood...')
t0 = time.time()

score = np.zeros((len(test_labels), k))

# Tính log Pr(label | image) bằng cách vector hóa
for label in range(k):
    rv = multivariate_normal(mean=mu[label], cov=sigma[label])
    # Compute log-pdf cho toàn bộ 10,000 ảnh test cùng lúc
    score[:, label] = np.log(pi[label]) + rv.logpdf(test_data)

# Chọn class có xác suất hậu định (posterior probability) lớn nhất
test_predictions = np.argmax(score, axis=1)

print(f'-> Đã dự đoán xong trong: {time.time() - t0:.2f} giây')


# ==========================================
# 4. KẾT QUẢ
# ==========================================
errors = np.sum(test_predictions != test_labels)
t_accuracy = np.mean(test_predictions == test_labels)

print('\n==========================================')
print(f'Số ảnh đoán sai: {errors} / 10,000')
print(f'Độ chính xác (Accuracy): {t_accuracy * 100:.2f}%')
print('==========================================')