import os
import urllib.request
import cv2
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
# 2. KHỞI TẠO BỘ PHÁT HIỆN HOG + LINEAR SVM
# ==========================================
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ==========================================
# 3. CHẠY DETECTMULTISCALE VỚI GROUPTHRESHOLD = 0
# ==========================================
# groupThreshold = 0: Giữ lại toàn bộ khung bao thô (chưa qua NMS/Gộp nhóm)
(foundBoundingBoxes, weights) = hog.detectMultiScale(
    img,
    hitThreshold=0,
    winStride=(4, 4),
    padding=(8, 8),
    scale=1.02,
    groupThreshold=0,
)

print(
    f'Số lượng khung bao thô phát hiện được (chưa qua NMS): {len(foundBoundingBoxes)}'
)

# ==========================================
# 4. VẼ KHUNG BAO VÀ TRỰC QUAN HÓA
# ==========================================
imgWithRawBboxes = img.copy()
for hx, hy, hw, hh in foundBoundingBoxes:
    cv2.rectangle(imgWithRawBboxes, (hx, hy), (hx + hw, hy + hh), (0, 0, 255), 1)

# Chuyển đổi màu BGR -> RGB để Matplotlib hiển thị đúng
imgWithRawBboxes = cv2.cvtColor(imgWithRawBboxes, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(imgWithRawBboxes)
plt.title(
    f'Khung bao thô HOG-SVM (Chưa lọc trùng lặp - Tổng số: {len(foundBoundingBoxes)})',
    fontsize=14,
    fontweight='bold',
)
plt.axis('off')
plt.tight_layout()
plt.show()