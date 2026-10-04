import gzip
import os
import time
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
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


# Nạp toàn bộ tập Train và Test
print('--- ĐANG NẠP DỮ LIỆU MNIST ---')
train_data = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')
test_data = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')


# ==========================================
# 2. TẠO DỰ ĐOÁN (GAUSSIAN BAYES MODEL)
# ==========================================
print('Đang huấn luyện mô hình và tạo dự đoán trên tập test...')
k, d = 10, train_data.shape[1]
mu, sigma, pi = np.zeros((k, d)), np.zeros((k, d, d)), np.zeros(k)
c = 3500  # Smoothing factor

for label in range(k):
    indices = train_labels == label
    pi[label] = np.sum(indices) / float(len(train_labels))
    mu[label] = np.mean(train_data[indices, :], axis=0)
    sigma[label] = np.cov(train_data[indices, :], rowvar=0, bias=1) + c * np.eye(d)

# Dự đoán nhãn cho 10,000 ảnh test
score = np.zeros((len(test_labels), k))
for label in range(k):
    rv = multivariate_normal(mean=mu[label], cov=sigma[label])
    score[:, label] = np.log(pi[label]) + rv.logpdf(test_data)

test_predictions = np.argmax(score, axis=1)


# ==========================================
# 3. ĐỊNH NGHĨA HÀM HIỂN THỊ ÂNH
# ==========================================
def display_char(image, title=''):
    plt.figure(figsize=(3, 3))
    plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
    if title:
        plt.title(title, fontsize=11, fontweight='bold')
    plt.axis('off')
    plt.show()


# ==========================================
# 4. LỌC VÀ HIỂN THỊ CÁC ẢNH BỊ PHÂN LOẠI SAI
# ==========================================
# Tìm vị trí các ảnh đoán sai
wrong_indices = test_predictions != test_labels

wrong_digits = test_data[wrong_indices]
wrong_preds = test_predictions[wrong_indices]
correct_labs = test_labels[wrong_indices]

print('\n==========================================')
print(f'Tổng số ảnh phân loại sai: {len(wrong_preds)} / {len(test_labels)}')
print('==========================================\n')

# Hiển thị ảnh phân loại sai tại chỉ số 1
title_text = f'Predicted: {wrong_preds[1]} | Actual: {correct_labs[1]}'
print(f'Hiển thị mẫu sai thứ 2 (chỉ số 1): {title_text}')
display_char(wrong_digits[1], title=title_text)

# Hiển thị thêm danh sách 5 ảnh phân loại sai đầu tiên
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
fig.suptitle('Top 5 ảnh bị dự đoán sai', fontsize=14, fontweight='bold')

for i in range(5):
    axes[i].imshow(np.reshape(wrong_digits[i], (28, 28)), cmap=plt.cm.gray)
    axes[i].set_title(
        f'Pred: {wrong_preds[i]}\nActual: {correct_labs[i]}', fontsize=10
    )
    axes[i].axis('off')

plt.tight_layout()
plt.show()