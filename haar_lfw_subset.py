from time import time
from dask import delayed
import matplotlib.pyplot as plt
import numpy as np
from skimage.data import lfw_subset
from skimage.feature import (
    draw_haar_like_feature,
    haar_like_feature,
    haar_like_feature_coord,
)
from skimage.transform import integral_image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# ==========================================
# 1. ĐỊNH NGHĨA HÀM TRÍCH XUẤT ĐẶC TRƯNG HAAR-LIKE (DASK DELAYED)
# ==========================================
@delayed
def extract_feature_image(img, feature_type, feature_coord=None):
    """Trích xuất đặc trưng Haar-like từ ma trận tích lũy (integral image)"""
    ii = integral_image(img)
    return haar_like_feature(
        ii,
        0,
        0,
        ii.shape[0],
        ii.shape[1],
        feature_type=feature_type,
        feature_coord=feature_coord,
    )


# ==========================================
# 2. NẠP VÀ KIỂM TRA DỮ LIỆU LFW SUBSET
# ==========================================
print('--- ĐANG NẠP TẬP DỮ LIỆU LFW SUBSET ---')
images = lfw_subset()
print('Kích thước mảng dữ liệu ảnh:', images.shape)


# ==========================================
# 3. TRỰC QUAN HÓA 25 KHUÔN MẶT ĐẦU TIÊN
# ==========================================
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=0.9, bottom=0, top=0.9, hspace=0.05, wspace=0.05
)

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(images[i, :, :], cmap='bone')
    plt.axis('off')

plt.suptitle('Faces (LFW Subset Dataset)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()