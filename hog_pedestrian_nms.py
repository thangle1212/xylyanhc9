import os
import urllib.request
import cv2
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. TẢI VÀ NẠP ẢNH MẪU
# ==========================================
image_path = 'pedestrians_sample.jpg'
if not os.path.exists(image_path):
    print('Đang tải ảnh mẫu người đi bộ...')
    url = 'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg'
    req = urllib.request.Request(
        url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with (
        urllib.request.urlopen(req, timeout=10) as response,
        open(image_path, 'wb') as out_file,
    ):
        out_file.write(response.read())

img = cv2.imread(image_path)

# ==========================================
# 2. KHỞI TẠO HOG DESCRIPTOR VÀ TRÍCH XUẤT KHUNG BAO THÔ
# ==========================================
print('Đang quét ảnh bằng HOG-SVM...')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Phát hiện khung bao thô (groupThreshold=0)
(foundBoundingBoxes, weights) = hog.detectMultiScale(
    img,
    hitThreshold=0,
    winStride=(4, 4),
    padding=(8, 8),
    scale=1.02,
    groupThreshold=0,
)

# ==========================================
# 3. LỌC TRÙNG LẶP BẰNG NON-MAXIMUM SUPPRESSION (NMS)
# ==========================================
# Chuyển định dạng từ (x, y, w, h) sang (x1, y1, x2, y2)
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in foundBoundingBoxes])

# Áp dụng NMS với ngưỡng đè lấp (overlap threshold) = 65%
nmsBoundingBoxes = non_max_suppression(rects, probs=None, overlapThresh=0.65)

print('\n==========================================')
print(f'Số lượng khung bao ban đầu (Thô)   : {len(rects)}')
print(f'Số lượng khung bao sau khi lọc NMS : {len(nmsBoundingBoxes)}')
print('==========================================\n')

# ==========================================
# 4. VẼ KHUNG BAO ĐÃ LỌC VÀ TRỰC QUAN HÓA
# ==========================================
img_nms = img.copy()
for x1, y1, x2, y2 in nmsBoundingBoxes:
    cv2.rectangle(img_nms, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Chuyển BGR sang RGB để Matplotlib hiển thị đúng màu
img_nms_rgb = cv2.cvtColor(img_nms, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(img_nms_rgb)
plt.title(
    f'Kết quả sau Non-Maximum Suppression (NMS) - Còn lại: {len(nmsBoundingBoxes)} khung bao',
    fontsize=14,
    fontweight='bold',
)
plt.axis('off')
plt.tight_layout()
plt.show()