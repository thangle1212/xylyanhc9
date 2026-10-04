import gzip
import os
import time
from urllib.request import urlretrieve
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# ==========================================
# 1. HÀM NẠP DỮ LIỆU
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
print('--- ĐANG NẠP DỮ LIỆU MNIST ---')
train_data = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')
test_data = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')


# ==========================================
# 2. BƯỚC TỐI ƯU 1: BẢO TOÀN THÔNG TIN & GIẢM CHIỀU BẰNG PCA
# ==========================================
print('\n[1/2] Đang nén dữ liệu từ 784 chiều xuống 50 chiều bằng PCA...')
t0 = time.time()

# Nén dữ liệu xuống 50 chiều (giữ lại hơn 85% năng lượng/thông tin của ảnh)
pca = PCA(n_components=50, random_state=42)
train_data_pca = pca.fit_transform(train_data)
test_data_pca = pca.transform(test_data)

print(f'-> Hoàn tất nén PCA trong: {time.time() - t0:.2f} giây')


# ==========================================
# 3. BƯỚC TỐI ƯU 2: PHÂN LOẠI 1-NN ĐA NHÂN CPU (n_jobs=-1)
# ==========================================
print('\n[2/2] Đang chạy 1-NN với tất cả nhân CPU (n_jobs=-1)...')
t0 = time.time()

# n_jobs=-1 cho phép kích hoạt tất cả các lõi CPU trên máy tính để tính toán song song
knn = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
knn.fit(train_data_pca, train_labels)

# Dự đoán 10,000 ảnh test
test_predictions = knn.predict(test_data_pca)
t_classify = time.time() - t0

print(f'-> Phân loại xong 10,000 ảnh test trong: {t_classify:.2f} giây')


# ==========================================
# 4. ĐÁNH GIÁ ĐỘ CHÍNH XÁC
# ==========================================
acc = accuracy_score(test_labels, test_predictions)
print(f'\n==========================================')
print(f'Độ chính xác trên tập test (Accuracy): {acc * 100:.2f}%')
print(f'==========================================')