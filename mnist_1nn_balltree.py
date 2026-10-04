import gzip
import os
import time
from urllib.request import urlretrieve
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import BallTree


# ==========================================
# 1. HÀM TẢI & NẠP DỮ LIỆU MNIST
# ==========================================
def download(filename, source='https://ossci-datasets.s3.amazonaws.com/mnist/'):
    if not os.path.exists(filename):
        print(f'Đang tải {filename}...')
        urlretrieve(source + filename, filename)
        print(f'Đã tải xong {filename}!')


def load_mnist_images(filename):
    download(filename)
    with gzip.open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)


def load_mnist_labels(filename):
    download(filename)
    with gzip.open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


# ==========================================
# 2. TẢI TẬP TRAIN (60,000 MẪU) & TEST (10,000 MẪU)
# ==========================================
print("--- ĐANG TẢI & NẠP DỮ LIỆU MNIST ---")
train_data = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')
test_data = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

print(f"Số lượng tập Train: {train_data.shape[0]} mẫu (28x28 = 784 pixels)")
print(f"Số lượng tập Test : {test_data.shape[0]} mẫu\n")


# ==========================================
# 3. PHẦN HUẤN LUYỆN VÀ ĐÁNH GIÁ 1-NN (BALL TREE)
# ==========================================

# Xây dựng cấu trúc cây BallTree trên tập dữ liệu huấn luyện
print("Đang xây dựng cấu trúc dữ liệu BallTree...")
t_before = time.time()
ball_tree = BallTree(train_data)
t_after = time.time()

t_training = t_after - t_before
print(f'Thời gian tạo BallTree: {t_training:.2f} giây')

# Tìm láng giềng gần nhất (k=1) cho 10,000 mẫu trong tập kiểm thử
print("\nĐang tìm láng giềng gần nhất (1-NN) cho 10,000 ảnh test...")
t_before = time.time()
test_neighbors = np.squeeze(
    ball_tree.query(test_data, k=1, return_distance=False)
)
test_predictions = train_labels[test_neighbors]
t_after = time.time()

t_testing = t_after - t_before
print(f'Thời gian phân loại tập test: {t_testing:.2f} giây')

# Tính độ chính xác phân loại (Accuracy)
acc = accuracy_score(test_labels, test_predictions)
print(f'\n==========================================')
print(f'Độ chính xác trên tập test (Accuracy): {acc * 100:.2f}%')
print(f'==========================================')