import matplotlib.pyplot as plt
from skimage.data import lfw_subset

# ==========================================
# 1. NẠP TẬP DỮ LIỆU LFW SUBSET
# ==========================================
print('--- ĐANG NẠP DỮ LIỆU LFW SUBSET ---')
images = lfw_subset()
print('Kích thước mảng dữ liệu ảnh:', images.shape)

# ==========================================
# 2. TRỰC QUAN HÓA 25 MẪU NON-FACES (CHỈ SỐ 100 - 124)
# ==========================================
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=0.9, bottom=0, top=0.9, hspace=0.05, wspace=0.05
)

for i in range(100, 125):
    plt.subplot(5, 5, i - 99)
    plt.imshow(images[i, :, :], cmap='bone')
    plt.axis('off')

plt.suptitle('Non-Faces (Chỉ số 100-124)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()