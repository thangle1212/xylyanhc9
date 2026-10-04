from time import time
import dask
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
from sklearn.model_selection import train_test_split


# ==========================================
# 1. NẠP DỮ LIỆU VÀ ĐỊNH NGHĨA HÀM DASK DELAYED
# ==========================================
print('--- ĐANG NẠP DỮ LIỆU LFW SUBSET ---')
images = lfw_subset()


@delayed
def extract_feature_image(img, feature_type, feature_coord=None):
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
# 2. TRÍCH XUẤT ĐẶC TRƯNG VÀ HUẤN LUYỆN MODEL
# ==========================================
feature_types = ['type-2-x', 'type-2-y']
X_delayed = [extract_feature_image(img, feature_types) for img in images]

print('Đang trích xuất đặc trưng Haar-like bằng Dask...')
X = np.array(dask.compute(*X_delayed, scheduler='threads'))

y = np.array([1] * 100 + [0] * 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=150, random_state=0, stratify=y
)

feature_coord, feature_type = haar_like_feature_coord(
    width=images.shape[2], height=images.shape[1], feature_type=feature_types
)

print('Đang huấn luyện mô hình Random Forest...')
clf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    max_features=100,
    n_jobs=-1,
    random_state=0,
)
clf.fit(X_train, y_train)


# ==========================================
# 3. TRỰC QUAN HÓA TOP 25 ĐẶC TRƯNG QUAN TRỌNG
# ==========================================
print('Đang vẽ Top 25 đặc trưng Haar-like quan trọng nhất...')

# Sắp xếp chỉ số các đặc trưng theo thứ tự độ quan trọng giảm dần (feature_importances_)
idx_sorted = np.argsort(clf.feature_importances_)[::-1]

fig, axes = plt.subplots(5, 5, figsize=(10, 10))

for idx, ax in enumerate(axes.ravel()):
    image = images[1].copy()  # Dùng ảnh khuôn mặt thứ 2 làm mẫu
    image_drawn = draw_haar_like_feature(
        image,
        0,
        0,
        images.shape[2],
        images.shape[1],
        [feature_coord[idx_sorted[idx]]],
    )
    ax.imshow(image_drawn)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    'Top 25 đặc trưng Haar-like quan trọng nhất', fontsize=18, y=0.95
)
plt.tight_layout()
plt.show()