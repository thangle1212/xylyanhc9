import gzip
import os
import time
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn.svm import SVC


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


# Nạp dữ liệu Train (60,000) và Test (10,000)
print('--- ĐANG NẠP DỮ LIỆU MNIST ---')
train_data = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')
test_data = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

# CHUẨN HÓA DỮ LIỆU: Đưa giá trị pixel từ [0, 255] về [0.0, 1.0] (Rất quan trọng cho SVM)
print('Đang chuẩn hóa giá trị pixel về dải [0, 1]...')
train_data_scaled = train_data / 255.0
test_data_scaled = test_data / 255.0


# ==========================================
# 2. HUẤN LUYỆN MÔ HÌNH SVM (POLY DEGREE 2)
# ==========================================
clf = SVC(C=1, kernel='poly', degree=2)

print('\nĐang huấn luyện mô hình SVM (Polynomial Kernel Degree 2)...')
t_start = time.time()
clf.fit(train_data_scaled, train_labels)
print(f'-> Hoàn tất huấn luyện trong: {time.time() - t_start:.2f} giây.')


# ==========================================
# 3. ĐÁNH GIÁ TRÊN TẬP TEST
# ==========================================
print('\nĐang dự đoán nhãn cho 10,000 ảnh test...')
t_pred = time.time()
test_predictions = clf.predict(test_data_scaled)
print(f'-> Đã dự đoán xong trong: {time.time() - t_pred:.2f} giây.')

# Tính độ chính xác (Accuracy)
accuracy = metrics.accuracy_score(test_labels, test_predictions)
print('\n==========================================')
print(f'Độ chính xác (Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)')
print('==========================================\n')


# ==========================================
# 4. TRỰC QUAN HÓA MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
# ==========================================
cm = metrics.confusion_matrix(test_labels, test_predictions)
df_cm = pd.DataFrame(cm, index=range(10), columns=range(10))

plt.figure(figsize=(10, 8))
sn.set_theme(font_scale=1.2)
sn.heatmap(df_cm, annot=True, annot_kws={'size': 14}, fmt='g', cmap='Blues')

plt.xlabel('Predicted Label (Nhãn dự đoán)', fontsize=12, fontweight='bold')
plt.ylabel('True Label (Nhãn thực tế)', fontsize=12, fontweight='bold')
plt.title(
    'Confusion Matrix cho mô hình SVM (Poly Kernel Degree 2) trên MNIST',
    fontsize=13,
    fontweight='bold',
)
plt.tight_layout()
plt.show()