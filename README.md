# Phòng thí nghiệm Machine Learning

Ứng dụng Streamlit gom các bài thực hành MNIST, PCA/Eigenfaces, phân cụm ảnh, HOG pedestrian và Haar-LFW. Các script gốc được giữ lại để đối chiếu; không import trực tiếp vì phần lớn script thực thi ngay khi import.

## Chạy ứng dụng

Mở PowerShell tại thư mục dự án và chạy:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

MNIST dùng bốn file `.gz` đã có trong thư mục. Nếu thiếu file, ứng dụng sẽ tải từ máy chủ MNIST. Olivetti Faces và LFW subset được tải lần đầu khi mở trang tương ứng; cần kết nối Internet cho lần tải đầu đó.

## Các phần trên web

| Phần | Chức năng | Script gốc |
| --- | --- | --- |
| MNIST | Gallery; BallTree 1-NN; PCA + KDTree; SVM; Gaussian Bayes; confusion matrix; ảnh dự đoán sai | `mnist_display.py`, `mnist_1nn_balltree.py`, `mnist_1nn_optimized.py`, `mnist_svm.py`, `mnist_bayes_eval.py`, `mnist_generative_model.py`, `mnist_confusion_matrix.py`, `mnist_misclassified.py` |
| PCA & khuôn mặt | Digits PCA 2D; Olivetti gallery; phương sai; Mean/SD Face; Eigenfaces; khai triển tuyến tính; tái tạo ảnh | `pca1.py`, `eigenface.py`, `eigenface1.py`, `eigenfaces.py`, `eigen_decomposition.py`, `reconstructed_compares.py`, `reconstructed_faces.py` |
| Ảnh & người đi bộ | K-Means; Spectral clustering; giảm màu; HOG raw boxes; NMS | `compare1.py`, `k-means.py`, `hog_pedestrian_raw.py`, `hog_pedestrian_nms.py` |
| Haar-LFW | Face/non-face gallery; Haar features; Random Forest; feature importance | `haar_lfw_subset.py`, `lfw_non_faces.py`, `haar_rf_lfw.py`, `visualize_top_features.py` |
| Kho script | Đọc source đầy đủ và xem nhóm/chức năng của từng file | Toàn bộ 23 script gốc |

Haar-LFW lấy mẫu số lượng Haar features và số cây để thời gian chạy phù hợp với giao diện tương tác. Các mô hình MNIST có thể giới hạn số mẫu train/test để không phải chờ chạy toàn bộ 60.000/10.000 mẫu.

## Tên file

Tên Python hiện tại được giữ nguyên để các tham chiếu cũ không bị ảnh hưởng. Nếu muốn chuẩn hóa thêm, có thể đổi `compare1.py` thành `image_segmentation_comparison.py`, `eigenface1.py` thành `olivetti_pca_statistics.py`, `eigen_decomposition.py` thành `olivetti_eigenface_decomposition.py`, `k-means.py` thành `image_color_quantization.py`, và `pca1.py` thành `digits_pca_projection.py`. Các trang thuật toán không import các script gốc; nếu đổi tên, hãy cập nhật `SCRIPT_INFO` trong `app.py` để nhãn trong Kho script tiếp tục chính xác.