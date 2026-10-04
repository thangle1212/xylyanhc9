import gzip
import os
import time
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn.neighbors import BallTree


# ==========================================
# 1. HÀM NẠP DỮ LIỆU MNIST
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


# Nạp tập dữ liệu Train (60,000) và Test (10,000)
print('--- ĐANG NẠP DỮ LIỆU MNIST ---')
train_data = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')
test_data = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')


# ==========================================
# 2. KHỞI TẠO BẢN DỰ ĐOÁN (DÙNG 1,000 ẢNH TEST)
# ==========================================
print('Đang khởi tạo mô hình BallTree để dự đoán 1,000 ảnh test...')
t0 = time.time()

ball_tree = BallTree(train_data)
test_data_sub = test_data[:1000]

# Tìm láng giềng gần nhất (k=1)
test_neighbors = np.squeeze(
    ball_tree.query(test_data_sub, k=1, return_distance=False)
)
test_predictions = train_labels[test_neighbors]

print(f'Dự đoán xong trong: {time.time() - t0:.2f} giây')


# ==========================================
# 3. TÍNH ĐỘ CHÍNH XÁC (ACCURACY)
# ==========================================
y_true = test_labels[: len(test_predictions)]
t_accuracy = sum(test_predictions == y_true) / float(len(y_true))

print('\n==========================================')
print(f'Độ chính xác (Accuracy): {t_accuracy:.4f} ({t_accuracy * 100:.2f}%)')
print('==========================================\n')


# ==========================================
# 4. TRỰC QUAN HÓA MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
# ==========================================
# Tạo Confusion Matrix
cm = metrics.confusion_matrix(y_true, test_predictions)
df_cm = pd.DataFrame(cm, index=range(10), columns=range(10))

# Vẽ Heatmap bằng Seaborn
plt.figure(figsize=(10, 8))
sn.set_theme(font_scale=1.2)
sn.heatmap(df_cm, annot=True, annot_kws={'size': 14}, fmt='g', cmap='Blues')

plt.xlabel('Predicted Label (Nhãn dự đoán)', fontsize=12, fontweight='bold')
plt.ylabel('True Label (Nhãn thực tế)', fontsize=12, fontweight='bold')
plt.title(
    'Confusion Matrix cho mô hình 1-NN trên MNIST (1,000 mẫu)',
    fontsize=14,
    fontweight='bold',
)
plt.tight_layout()
plt.show()