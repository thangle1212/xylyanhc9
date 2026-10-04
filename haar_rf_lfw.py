from time import time
import dask
from dask import delayed
import numpy as np
from skimage.data import lfw_subset
from skimage.feature import haar_like_feature, haar_like_feature_coord
from skimage.transform import integral_image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. NẠP DỮ LIỆU VÀ ĐỊNH NGHĨA HÀM DASK DELAYED
# ==========================================
print('--- ĐANG NẠP DỮ LIỆU LFW SUBSET ---')
images = lfw_subset()
print(f'Kích thước mảng dữ liệu ảnh: {images.shape}')


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
# 2. TRÍCH XUẤT ĐẶC TRƯNG HAAR-LIKE SONG SONG (DASK)
# ==========================================
feature_types = ['type-2-x', 'type-2-y']

# Tạo đồ thị tính toán Dask (Parallel Computation Graph)
X_delayed = [extract_feature_image(img, feature_types) for img in images]

print('\nĐang trích xuất đặc trưng Haar-like song song bằng Dask...')
t_start = time()
X = np.array(dask.compute(*X_delayed, scheduler='threads'))
time_full_feature_comp = time() - t_start

print(f'-> Thời gian trích xuất đặc trưng: {time_full_feature_comp:.2f} giây')
print(f'-> Kích thước toàn bộ đặc trưng X: {X.shape}')


# ==========================================
# 3. CHUẨN BỊ NHÃN VÀ CHIA TẬP TRAIN / TEST
# ==========================================
# 100 mẫu mặt (1), 100 mẫu không phải mặt (0)
y = np.array([1] * 100 + [0] * 100)

# Chia tập dữ liệu: 150 train / 50 test theo tỷ lệ nhãn đều (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=150, random_state=0, stratify=y
)

print(f'-> Kích thước tập Train X_train: {X_train.shape}')
print(f'-> Kích thước tập Test X_test  : {X_test.shape}')


# ==========================================
# 4. TRÍCH XUẤT TỌA ĐỘ VÀ LOẠI ĐẶC TRƯNG
# ==========================================
feature_coord, feature_type = haar_like_feature_coord(
    width=images.shape[2], height=images.shape[1], feature_type=feature_types
)


# ==========================================
# 5. HUẤN LUYỆN VÀ ĐÁNH GIÁ RANDOM FOREST
# ==========================================
clf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    max_features=100,
    n_jobs=-1,
    random_state=0,
)

print('\nĐang huấn luyện Random Forest Classifier (1000 cây, n_jobs=-1)...')
t_start = time()
clf.fit(X_train, y_train)
time_full_train = time() - t_start

print(f'-> Thời gian huấn luyện: {time_full_train:.4f} giây')

# Tính điểm ROC-AUC trên tập kiểm thử
y_pred_proba = clf.predict_proba(X_test)[:, 1]
auc_full_features = roc_auc_score(y_test, y_pred_proba)

print('\n==========================================')
print(f'Chỉ số ROC-AUC (Full Features): {auc_full_features:.4f}')
print('==========================================')