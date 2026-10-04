from __future__ import annotations

import gzip
import time
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import dask
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dask import delayed
from scipy.linalg import cho_factor, solve_triangular
from skimage.data import lfw_subset
from skimage.feature import (
    draw_haar_like_feature,
    haar_like_feature,
    haar_like_feature_coord,
)
from skimage.transform import integral_image
from sklearn.cluster import MiniBatchKMeans, SpectralClustering
from sklearn.datasets import fetch_olivetti_faces, load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from PIL import Image
from io import BytesIO


ROOT = Path(__file__).resolve().parent
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

SCRIPT_INFO = {
    "compare1.py": ("Ảnh & phân cụm", "So sánh phân cụm K-Means và Spectral trên ảnh."),
    "eigenface.py": ("PCA & khuôn mặt", "Thư viện 25 khuôn mặt Olivetti."),
    "eigenface1.py": ("PCA & khuôn mặt", "Phương sai PCA, Mean Face và SD Face."),
    "eigenfaces.py": ("PCA & khuôn mặt", "Hiển thị các Eigenfaces."),
    "eigen_decomposition.py": ("PCA & khuôn mặt", "Phân tích ảnh thành các thành phần Eigenface."),
    "haar_lfw_subset.py": ("Haar-LFW", "Thư viện khuôn mặt LFW."),
    "haar_rf_lfw.py": ("Haar-LFW", "Đặc trưng Haar và Random Forest với ROC-AUC."),
    "hog_pedestrian_nms.py": ("Ảnh & người đi bộ", "HOG-SVM và lọc khung bằng NMS."),
    "hog_pedestrian_raw.py": ("Ảnh & người đi bộ", "Các khung phát hiện thô HOG-SVM."),
    "k-means.py": ("Ảnh & phân cụm", "Giảm màu ảnh bằng K-Means và chọn màu ngẫu nhiên."),
    "lfw_non_faces.py": ("Haar-LFW", "Thư viện mẫu non-face của LFW."),
    "mnist_1nn_balltree.py": ("MNIST", "Đánh giá 1-NN dùng BallTree."),
    "mnist_1nn_optimized.py": ("MNIST", "PCA và 1-NN dùng KDTree."),
    "mnist_bayes_eval.py": ("MNIST", "Gaussian Bayes đa biến và đánh giá."),
    "mnist_confusion_matrix.py": ("MNIST", "Confusion matrix cho BallTree 1-NN."),
    "mnist_display.py": ("MNIST", "Hiển thị ảnh và nhãn MNIST."),
    "mnist_generative_model.py": ("MNIST", "Mô hình sinh Gaussian và ảnh trung bình."),
    "mnist_misclassified.py": ("MNIST", "Trực quan hóa các chữ số dự đoán sai."),
    "mnist_svm.py": ("MNIST", "SVM polynomial và confusion matrix."),
    "pca1.py": ("PCA & khuôn mặt", "PCA và biểu đồ 2D trên chữ số 8x8."),
    "reconstructed_compares.py": ("PCA & khuôn mặt", "So sánh một ảnh Olivetti gốc/tái tạo."),
    "reconstructed_faces.py": ("PCA & khuôn mặt", "Tái tạo và xem nhiều khuôn mặt."),
    "visualize_top_features.py": ("Haar-LFW", "Trực quan hóa đặc trưng Haar quan trọng."),
}

RENAME_SUGGESTIONS = {
    "compare1.py": "image_segmentation_comparison.py",
    "eigenface1.py": "olivetti_pca_statistics.py",
    "eigen_decomposition.py": "olivetti_eigenface_decomposition.py",
    "k-means.py": "image_color_quantization.py",
    "pca1.py": "digits_pca_projection.py",
}

st.set_page_config(page_title="Phòng thí nghiệm ML", page_icon="ML", layout="wide")
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
      :root { --ink:#152a2b; --muted:#627473; --paper:#f5f7f2; --line:#dce4dc; --mint:#d8eee4; --coral:#df7356; }
    .stApp, [data-testid="stAppViewContainer"] { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
    [data-testid="stSidebar"] { background:#e9f0e9; border-right:1px solid var(--line); color:var(--ink) !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] span, [data-testid="stWidgetLabel"] p,
    [data-testid="stRadio"] label p { color:var(--ink) !important; }
      h1,h2,h3 { color:var(--ink); letter-spacing:0; }
      .eyebrow { color:#397c6b; font:500 12px 'DM Mono',monospace; text-transform:uppercase; }
      .hero { padding:1.2rem 0 1.6rem; border-bottom:1px solid var(--line); margin-bottom:1.2rem; }
      .hero h1 { font-size:2.4rem; margin:.2rem 0; }
      .hero p { color:var(--muted); max-width:760px; margin:0; }
      .stButton > button { border-radius:5px; border:1px solid #245b52; background:#245b52; color:white; }
      .stButton > button:hover { background:#347467; border-color:#347467; color:white; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:5px; padding:12px 14px; color:var(--ink) !important; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] *,
    [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * { color:var(--ink) !important; }
      code, pre { font-family:'DM Mono',monospace !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def show_scripts(names: list[str]) -> None:
    st.caption("Script gốc: " + " · ".join(f"`{name}`" for name in names))


@st.cache_data(show_spinner=False)
def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    for name, url in MNIST_FILES.items():
        path = ROOT / name
        if not path.exists():
            urlretrieve(url, path)
        with gzip.open(path, "rb") as stream:
            offset = 16 if "images" in name else 8
            values = np.frombuffer(stream.read(), dtype=np.uint8, offset=offset)
        if "images" in name:
            values = values.reshape(-1, 784)
        arrays.append(values)
    return tuple(arrays)  # type: ignore[return-value]


@st.cache_data(show_spinner=False)
def evaluate_mnist(model_name: str, train_size: int, test_size: int):
    train_x, train_y, test_x, test_y = load_mnist()
    train_x = train_x[:train_size]
    train_y = train_y[:train_size]
    test_x = test_x[:test_size]
    test_y = test_y[:test_size]
    started = time.perf_counter()

    if model_name == "BallTree 1-NN":
        neighbors = BallTree(train_x).query(test_x, k=1, return_distance=False).ravel()
        predictions = train_y[neighbors]
    elif model_name == "PCA + KDTree 1-NN":
        component_count = min(50, train_x.shape[0] - 1, train_x.shape[1])
        pca = PCA(n_components=component_count, svd_solver="randomized", random_state=42)
        train_proj = pca.fit_transform(train_x)
        test_proj = pca.transform(test_x)
        model = KNeighborsClassifier(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
        model.fit(train_proj, train_y)
        predictions = model.predict(test_proj)
    elif model_name == "SVM polynomial":
        model = SVC(C=1, kernel="poly", degree=2)
        model.fit(train_x / 255.0, train_y)
        predictions = model.predict(test_x / 255.0)
    else:
        predictions = _full_covariance_bayes(train_x, train_y, test_x)

    elapsed = time.perf_counter() - started
    matrix = confusion_matrix(test_y, predictions, labels=np.arange(10))
    wrong = np.flatnonzero(predictions != test_y)
    return predictions, test_y, matrix, wrong, elapsed, accuracy_score(test_y, predictions)


def _full_covariance_bayes(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    scores = np.empty((len(test_x), 10), dtype=np.float64)
    feature_count = train_x.shape[1]
    for label in range(10):
        class_x = train_x[train_y == label].astype(np.float64)
        if len(class_x) < 2:
            raise ValueError("Mỗi chữ số cần tối thiểu hai mẫu trong tập train.")
        mean = class_x.mean(axis=0)
        centered = class_x - mean
        covariance = centered.T @ centered / len(class_x)
        covariance.flat[:: feature_count + 1] += 3500.0
        factor, lower = cho_factor(covariance, lower=True, check_finite=False)
        root = np.tril(factor) if lower else np.triu(factor).T
        delta = (test_x.astype(np.float64) - mean).T
        whitened = solve_triangular(root, delta, lower=True, check_finite=False)
        log_det = 2.0 * np.log(np.diag(root)).sum()
        scores[:, label] = -0.5 * (
            np.einsum("ij,ij->j", whitened, whitened)
            + log_det
            + feature_count * np.log(2.0 * np.pi)
        )
    return np.argmax(scores, axis=1)


@st.cache_data(show_spinner=False)
def get_olivetti() -> np.ndarray:
    return fetch_olivetti_faces(shuffle=True, random_state=42).data


@st.cache_data(show_spinner=False)
def fit_olivetti_pca(component_count: int):
    faces = get_olivetti()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(faces)
    pca = PCA(n_components=component_count, svd_solver="randomized", random_state=42)
    projected = pca.fit_transform(scaled)
    return faces, scaler.mean_, np.maximum(scaler.scale_, 1e-8), projected, pca.components_, pca.explained_variance_ratio_


@st.cache_data(show_spinner=False)
def extract_haar_data(feature_count: int):
    images = lfw_subset()
    feature_types = ["type-2-x", "type-2-y"]
    coordinates, coordinate_types = haar_like_feature_coord(
        width=images.shape[2], height=images.shape[1], feature_type=feature_types
    )
    rng = np.random.default_rng(42)
    selected = np.sort(rng.choice(len(coordinates), size=min(feature_count, len(coordinates)), replace=False))
    coordinates = coordinates[selected]
    coordinate_types = coordinate_types[selected]

    @delayed
    def extract(image):
        ii = integral_image(image)
        return haar_like_feature(
            ii, 0, 0, ii.shape[1], ii.shape[0],
            feature_type=coordinate_types, feature_coord=coordinates
        )

    features = np.asarray(dask.compute(*(extract(image) for image in images), scheduler="threads"))
    labels = np.array([1] * 100 + [0] * 100)
    return images, features, labels, coordinates, coordinate_types


def plot_matrix(matrix: np.ndarray, title: str, cmap: str = "viridis") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, cmap=cmap, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thật")
    fig.tight_layout()
    return fig


def mnist_page() -> None:
    page_header("01 / MNIST", "Chữ số viết tay", "Xem dữ liệu, so sánh bộ phân loại và kiểm tra những dự đoán sai trong cùng một nơi.")
    show_scripts([name for name, (section, _) in SCRIPT_INFO.items() if section == "MNIST"])
    mode = st.radio("Thao tác", ["Ảnh mẫu", "Đánh giá mô hình", "Bayes sinh", "Ảnh dự đoán sai"], horizontal=True)

    try:
        train_x, train_y, test_x, test_y = load_mnist()
    except Exception as exc:
        st.error(f"Không nạp được MNIST: {exc}")
        return

    if mode == "Ảnh mẫu":
        st.caption(f"Train: {len(train_x):,} ảnh · Test: {len(test_x):,} ảnh · Mỗi ảnh 28 × 28 pixel")
        count = st.slider("Số ảnh hiển thị", 5, 25, 25, key="mnist_gallery_count")
        fig, axes = plt.subplots(int(np.ceil(count / 5)), 5, figsize=(9, 1.9 * int(np.ceil(count / 5))))
        for index, ax in enumerate(np.asarray(axes).ravel()):
            ax.axis("off")
            if index < count:
                ax.imshow(test_x[index].reshape(28, 28), cmap="gray")
                ax.set_title(f"Nhãn {test_y[index]}", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        return

    model_name = st.selectbox(
        "Mô hình",
        ["BallTree 1-NN", "PCA + KDTree 1-NN", "SVM polynomial", "Gaussian Bayes đa biến"],
    )
    max_train = min(20000, len(train_x))
    train_size = st.slider("Số mẫu train", 500, max_train, min(5000, max_train), step=500)
    test_size = st.slider("Số mẫu test", 100, min(3000, len(test_x)), min(500, len(test_x)), step=100)
    st.caption("Bayes dùng hiệp phương sai đa biến và regularization 3500; các mô hình khác giữ đúng hướng tiếp cận trong script.")
    run_label = "Huấn luyện và đánh giá" if mode != "Ảnh dự đoán sai" else "Chạy mô hình và tìm ảnh sai"
    if st.button(run_label, type="primary"):
        try:
            with st.spinner("Đang chạy trên tập con đã chọn..."):
                result = evaluate_mnist(model_name, train_size, test_size)
            st.session_state["mnist_result"] = result
            st.session_state["mnist_result_name"] = model_name
        except Exception as exc:
            st.error(f"Chạy mô hình thất bại: {exc}")

    result = st.session_state.get("mnist_result")
    if result is None:
        st.info("Chọn quy mô dữ liệu rồi chạy mô hình để xem kết quả.")
        if mode == "Bayes sinh":
            st.markdown("Ảnh trung bình theo lớp được tính từ các mẫu train, tương tự mô hình sinh trong script.")
            mean_fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            for digit, ax in enumerate(axes.ravel()):
                ax.imshow(train_x[train_y == digit].mean(axis=0).reshape(28, 28), cmap="gray")
                ax.set_title(f"Chữ số {digit}")
                ax.axis("off")
            st.pyplot(mean_fig)
            plt.close(mean_fig)
        return

    predictions, labels, matrix, wrong, elapsed, accuracy = result
    st.caption(f"Kết quả gần nhất: {st.session_state.get('mnist_result_name', model_name)}")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Accuracy", f"{accuracy:.2%}")
    col_b.metric("Ảnh phân loại sai", f"{len(wrong):,} / {len(labels):,}")
    col_c.metric("Thời gian", f"{elapsed:.2f} giây")

    if mode in ("Đánh giá mô hình", "Bayes sinh"):
        fig = plot_matrix(matrix, "Confusion matrix", "Blues")
        st.pyplot(fig)
        plt.close(fig)
    if mode == "Bayes sinh":
        means = np.stack([train_x[train_y == digit].mean(axis=0) for digit in range(10)])
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for digit, ax in enumerate(axes.ravel()):
            ax.imshow(means[digit].reshape(28, 28), cmap="gray")
            ax.set_title(f"Mean {digit}")
            ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    if mode == "Ảnh dự đoán sai":
        if not len(wrong):
            st.success("Không có ảnh sai trong tập test đã chọn.")
        else:
            count = min(20, len(wrong))
            fig, axes = plt.subplots(int(np.ceil(count / 5)), 5, figsize=(10, 2 * int(np.ceil(count / 5))))
            for index, ax in enumerate(np.asarray(axes).ravel()):
                ax.axis("off")
                if index < count:
                    sample = wrong[index]
                    ax.imshow(test_x[sample].reshape(28, 28), cmap="gray")
                    ax.set_title(f"Đoán {predictions[sample]} / Thật {labels[sample]}", fontsize=9)
            st.pyplot(fig)
            plt.close(fig)


def pca_page() -> None:
    page_header("02 / PCA", "Không gian khuôn mặt", "Khám phá phép chiếu PCA từ chữ số 8 × 8 đến Eigenfaces Olivetti và tái tạo ảnh.")
    show_scripts([name for name, (section, _) in SCRIPT_INFO.items() if section == "PCA & khuôn mặt"])
    dataset = st.radio("Tập dữ liệu", ["Olivetti Faces", "Digits 8 × 8"], horizontal=True)

    if dataset == "Digits 8 × 8":
        digits = load_digits()
        count = st.slider("Số chữ số trong gallery", 5, 25, 25)
        fig, axes = plt.subplots(5, 5, figsize=(7, 7))
        rng = np.random.default_rng(1)
        indices = rng.choice(len(digits.data), size=count, replace=False)
        for index, ax in enumerate(axes.ravel()):
            ax.axis("off")
            if index < count:
                sample = indices[index]
                ax.imshow(digits.images[sample], cmap="binary")
                ax.set_title(str(digits.target[sample]))
        st.pyplot(fig)
        plt.close(fig)
        if st.button("Chiếu dữ liệu xuống 2D"):
            pca = PCA(n_components=2)
            projected = pca.fit_transform(digits.data)
            fig, ax = plt.subplots(figsize=(9, 6))
            points = ax.scatter(projected[:, 0], projected[:, 1], c=digits.target, cmap="tab10", s=14, alpha=0.75)
            fig.colorbar(points, ax=ax, ticks=range(10), label="Chữ số")
            ax.set(xlabel="PC1", ylabel="PC2", title="PCA 2D trên Digits")
            ax.grid(alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)
            st.metric("Phương sai giữ lại với 2 thành phần", f"{pca.explained_variance_ratio_.sum():.1%}")
        return

    experiment = st.selectbox("Phân tích", ["Phương sai, Mean/SD và Eigenfaces", "Tái tạo ảnh", "Tổ hợp Eigenfaces"])
    components = st.slider("Số thành phần PCA", 5, 150, 64, key="olivetti_components")
    sample_index = st.slider("Ảnh Olivetti", 0, 399, 0)
    try:
        with st.spinner("Đang tải Olivetti và chạy PCA lần đầu..."):
            faces, mean, scale, projected, eigenfaces, variance = fit_olivetti_pca(components)
    except Exception as exc:
        st.error(f"Không tải được Olivetti Faces. Kiểm tra kết nối mạng rồi thử lại. Chi tiết: {exc}")
        return

    mean_face = mean.reshape(64, 64)
    sd_face = scale.reshape(64, 64)
    if experiment == "Phương sai, Mean/SD và Eigenfaces":
        left, right = st.columns([1.3, 1])
        with left:
            fig, ax = plt.subplots(figsize=(8, 4))
            cumulative = np.cumsum(variance)
            ax.plot(np.arange(1, len(cumulative) + 1), cumulative, color="#247566", linewidth=2)
            ax.set(xlabel="Số thành phần", ylabel="Phương sai tích lũy", ylim=(0, 1), title="Thông tin giữ lại")
            ax.grid(alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)
            st.metric(f"Thông tin với {components} PCs", f"{cumulative[-1]:.1%}")
        with right:
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(mean_face, cmap="bone"); axes[0].set_title("Mean Face"); axes[0].axis("off")
            axes[1].imshow(sd_face, cmap="bone"); axes[1].set_title("SD Face"); axes[1].axis("off")
            st.pyplot(fig)
            plt.close(fig)
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for index, ax in enumerate(axes.ravel()):
            ax.imshow(eigenfaces[index].reshape(64, 64), cmap="bone")
            ax.set_title(f"Eigenface {index + 1}")
            ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    else:
        restored_scaled = projected[sample_index] @ eigenfaces
        restored = (restored_scaled * scale + mean).reshape(64, 64)
        original = faces[sample_index].reshape(64, 64)
        if experiment == "Tổ hợp Eigenfaces":
            count = min(4, components)
            fig, axes = plt.subplots(1, count + 1, figsize=(3 * (count + 1), 3))
            axes[0].imshow(original, cmap="bone"); axes[0].set_title("Ảnh gốc"); axes[0].axis("off")
            for index in range(count):
                axes[index + 1].imshow(eigenfaces[index].reshape(64, 64), cmap="bone")
                axes[index + 1].set_title(f"PC {index + 1}\n{projected[sample_index, index]:+.3f}")
                axes[index + 1].axis("off")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(original, cmap="bone"); axes[0].set_title("Ảnh gốc"); axes[0].axis("off")
            axes[1].imshow(restored, cmap="bone"); axes[1].set_title(f"Tái tạo từ {components} PCs"); axes[1].axis("off")
        st.pyplot(fig)
        plt.close(fig)


def read_uploaded_image(uploaded, local_name: str) -> np.ndarray:
    if uploaded is not None:
        return np.asarray(Image.open(BytesIO(uploaded.getvalue())).convert("RGB"))
    return np.asarray(Image.open(ROOT / local_name).convert("RGB"))


def resize_image(image: np.ndarray, longest_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, longest_side / max(height, width))
    if scale == 1:
        return image
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def run_nms(boxes: list[tuple[int, int, int, int]], scores: list[float], threshold: float) -> list[int]:
    if not boxes:
        return []
    values = np.asarray(boxes, dtype=np.float32)
    x1, y1 = values[:, 0], values[:, 1]
    x2, y2 = x1 + values[:, 2], y1 + values[:, 3]
    areas = values[:, 2] * values[:, 3]
    order = np.argsort(scores)
    selected = []
    while order.size:
        current = int(order[-1])
        selected.append(current)
        order = order[:-1]
        if not order.size:
            break
        xx1 = np.maximum(x1[current], x1[order])
        yy1 = np.maximum(y1[current], y1[order])
        xx2 = np.minimum(x2[current], x2[order])
        yy2 = np.minimum(y2[current], y2[order])
        overlap = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1) / np.maximum(areas[order], 1)
        order = order[overlap <= threshold]
    return selected


def image_page() -> None:
    page_header("03 / THỊ GIÁC MÁY", "Phân cụm & người đi bộ", "Thử giảm màu, phân đoạn ảnh bằng K-Means/Spectral và phát hiện HOG-SVM trên ảnh của bạn.")
    names = [name for name, (section, _) in SCRIPT_INFO.items() if section in ("Ảnh & phân cụm", "Ảnh & người đi bộ")]
    show_scripts(names)
    task = st.radio("Tác vụ", ["Phân cụm / giảm màu", "HOG người đi bộ"], horizontal=True)
    samples = [name for name in ("pepper.jpg", "keanu-reeves.jpg", "pedestrians_sample.jpg", "me.jpg") if (ROOT / name).exists()]
    local_name = st.selectbox("Ảnh mẫu", samples)
    uploaded = st.file_uploader("Hoặc tải ảnh lên", type=["png", "jpg", "jpeg", "webp"])
    try:
        original = read_uploaded_image(uploaded, local_name)
    except Exception as exc:
        st.error(f"Không đọc được ảnh: {exc}")
        return

    if task == "Phân cụm / giảm màu":
        algorithm = st.selectbox("Thuật toán", ["K-Means", "Spectral", "So sánh cả hai"])
        clusters = st.slider("Số cụm màu", 2, 16, 4)
        max_side = 96 if algorithm in ("Spectral", "So sánh cả hai") else 240
        image = resize_image(original, max_side)
        pixels = image.reshape(-1, 3).astype(np.float32) / 255.0
        if st.button("Phân tích ảnh", type="primary"):
            results = []
            with st.spinner("Đang phân cụm các pixel..."):
                if algorithm in ("K-Means", "So sánh cả hai"):
                    model = MiniBatchKMeans(n_clusters=clusters, random_state=10, n_init=3, batch_size=2048)
                    labels = model.fit_predict(pixels)
                    quantized = model.cluster_centers_[labels].reshape(image.shape).clip(0, 1)
                    results.append(("K-Means", labels.reshape(image.shape[:2]), quantized))
                if algorithm in ("Spectral", "So sánh cả hai"):
                    spectral_image = resize_image(image, 72)
                    spectral_pixels = spectral_image.reshape(-1, 3).astype(np.float32) / 255.0
                    model = SpectralClustering(
                        n_clusters=clusters, affinity="nearest_neighbors", n_neighbors=min(10, len(spectral_pixels) - 1),
                        eigen_solver="arpack", random_state=10, n_jobs=-1,
                    )
                    labels = model.fit_predict(spectral_pixels).reshape(spectral_image.shape[:2])
                    results.append(("Spectral", labels, None))
            st.session_state["image_cluster_result"] = (image, results)
        result = st.session_state.get("image_cluster_result")
        if result:
            image, results = result
            st.image(original, caption="Ảnh gốc", width="stretch")
            cols = st.columns(len(results))
            for column, (name, labels, quantized) in zip(cols, results):
                with column:
                    if quantized is not None:
                        st.image(quantized, caption=f"{name} · ảnh giảm màu", width="stretch")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.imshow(labels, cmap="tab20"); ax.set_title(f"Phân đoạn {name}"); ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)
    else:
        stride = st.slider("Bước quét", 4, 16, 8, step=4)
        overlap_threshold = st.slider("Ngưỡng NMS", 0.1, 0.9, 0.65, step=0.05)
        image = resize_image(original, 900)
        if st.button("Quét người đi bộ", type="primary"):
            with st.spinner("Đang quét ảnh bằng HOG-SVM..."):
                detector_input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                found, weights = hog.detectMultiScale(
                    detector_input, hitThreshold=0, winStride=(stride, stride), padding=(8, 8), scale=1.05, groupThreshold=0
                )
                boxes = [tuple(map(int, box)) for box in found]
                scores = np.asarray(weights).reshape(-1).astype(float).tolist()
                kept = run_nms(boxes, scores, overlap_threshold)
                raw_image = image.copy()
                filtered_image = image.copy()
                for x, y, width, height in boxes:
                    cv2.rectangle(raw_image, (x, y), (x + width, y + height), (231, 91, 67), 1)
                for index in kept:
                    x, y, width, height = boxes[index]
                    cv2.rectangle(filtered_image, (x, y), (x + width, y + height), (38, 137, 92), 2)
                st.session_state["hog_result"] = (raw_image, filtered_image, len(boxes), len(kept))
        result = st.session_state.get("hog_result")
        if result:
            raw_image, filtered_image, raw_count, kept_count = result
            first, second = st.columns(2)
            first.metric("Khung thô", raw_count)
            second.metric("Sau NMS", kept_count)
            first, second = st.columns(2)
            first.image(raw_image, caption="HOG-SVM · khung thô", width="stretch")
            second.image(filtered_image, caption="Sau Non-Maximum Suppression", width="stretch")


def haar_page() -> None:
    page_header("04 / HAAR + LFW", "Đặc trưng khuôn mặt", "Xem ảnh LFW, huấn luyện Random Forest từ Haar-like features và trực quan hóa đặc trưng quan trọng.")
    show_scripts([name for name, (section, _) in SCRIPT_INFO.items() if section == "Haar-LFW"])
    task = st.radio("Thao tác", ["Thư viện LFW", "Phân loại Haar + Random Forest"], horizontal=True)
    try:
        images = lfw_subset()
    except Exception as exc:
        st.error(f"Không tải được LFW subset: {exc}")
        return
    if task == "Thư viện LFW":
        kind = st.radio("Mẫu ảnh", ["Khuôn mặt", "Non-face"], horizontal=True)
        start = 0 if kind == "Khuôn mặt" else 100
        fig, axes = plt.subplots(5, 5, figsize=(7, 7))
        for index, ax in enumerate(axes.ravel()):
            ax.imshow(images[start + index], cmap="bone")
            ax.axis("off")
        fig.suptitle(f"25 mẫu {kind} từ LFW")
        st.pyplot(fig)
        plt.close(fig)
        return

    feature_count = st.slider("Số Haar features lấy mẫu", 100, 1200, 400, step=100)
    tree_count = st.slider("Số cây Random Forest", 50, 400, 150, step=50)
    if st.button("Huấn luyện bộ phân loại", type="primary"):
        try:
            with st.spinner("Đang trích xuất Haar features và huấn luyện..."):
                images, features, labels, coordinates, coordinate_types = extract_haar_data(feature_count)
                train_x, test_x, train_y, test_y = train_test_split(
                    features, labels, train_size=150, random_state=0, stratify=labels
                )
                model = RandomForestClassifier(
                    n_estimators=tree_count, max_features=min(100, features.shape[1]), n_jobs=-1, random_state=0
                )
                model.fit(train_x, train_y)
                predictions = model.predict(test_x)
                st.session_state["haar_result"] = (
                    images, coordinates, coordinate_types, model, test_y, predictions,
                    accuracy_score(test_y, predictions),
                )
        except Exception as exc:
            st.error(f"Huấn luyện Haar thất bại: {exc}")
    result = st.session_state.get("haar_result")
    if result:
        images, coordinates, coordinate_types, model, test_y, predictions, accuracy = result
        st.metric("Accuracy trên test", f"{accuracy:.2%}")
        fig = plot_matrix(confusion_matrix(test_y, predictions, labels=[0, 1]), "LFW · Non-face / Face", "Greens")
        st.pyplot(fig)
        plt.close(fig)
        important = np.argsort(model.feature_importances_)[::-1][:15]
        fig, axes = plt.subplots(3, 5, figsize=(10, 6))
        for index, ax in zip(important, axes.ravel()):
            drawn = draw_haar_like_feature(
                images[1].copy(), 0, 0, images.shape[2], images.shape[1],
                [coordinates[index]],
            )
            ax.imshow(drawn)
            ax.set_title(f"#{index + 1} · {model.feature_importances_[index]:.3f}", fontsize=9)
            ax.axis("off")
        fig.suptitle("15 Haar features quan trọng nhất")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Script gốc dùng toàn bộ Haar features và 1.000 cây. Ứng dụng lấy mẫu số features/cây để thời gian chạy phù hợp khi tương tác.")


def catalog_page() -> None:
    page_header("05 / SOURCE", "Kho script Python", "Toàn bộ script gốc được giữ nguyên; xem nhanh nội dung và đối chiếu với phần tương ứng trên web.")
    files = sorted(path for path in ROOT.glob("*.py") if path.name != Path(__file__).name)
    if not files:
        st.info("Không tìm thấy script Python trong thư mục dự án.")
        return
    selected_name = st.selectbox("Chọn file", [path.name for path in files])
    selected = ROOT / selected_name
    section, description = SCRIPT_INFO.get(selected_name, ("Ứng dụng", "Ứng dụng Streamlit."))
    st.markdown(f"**Phần:** {section}  \n**Nội dung:** {description}")
    if selected_name in RENAME_SUGGESTIONS:
        st.caption(f"Tên gợi ý nếu muốn đổi: `{RENAME_SUGGESTIONS[selected_name]}`. Tên hiện tại vẫn được giữ để tránh làm hỏng tham chiếu cũ.")
    st.code(selected.read_text(encoding="utf-8"), language="python", line_numbers=True)
    st.caption(f"Đã lập danh mục {len(SCRIPT_INFO)} script gốc. Các script tự chạy khi import, vì vậy ứng dụng gọi lại thuật toán an toàn thay vì import trực tiếp.")


with st.sidebar:
    st.markdown("<div class='eyebrow'>LAB / COMPUTER VISION</div>", unsafe_allow_html=True)
    st.markdown("## Điều hướng")
    page = st.radio(
        "Khu vực",
        ["Tổng quan", "MNIST", "PCA & khuôn mặt", "Ảnh & người đi bộ", "Haar-LFW", "Kho script"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Dữ liệu MNIST dùng file `.gz` có sẵn. Olivetti/LFW sẽ tải lần đầu khi mở phần tương ứng.")


if page == "Tổng quan":
    page_header("MACHINE LEARNING / COMPUTER VISION", "Phòng thí nghiệm", "Một giao diện để xem dữ liệu, chạy thuật toán và đối chiếu kết quả từ các bài thực hành Python.")
    cards = st.columns(4)
    cards[0].metric("Script đã gom", len(SCRIPT_INFO))
    cards[1].metric("Nhóm thực hành", 4)
    cards[2].metric("Dữ liệu MNIST", "Có sẵn")
    cards[3].metric("File cần chạy", "app.py")
    st.markdown("### Chọn một phòng thực hành")
    overview = st.columns(4)
    summaries = [
        ("01", "MNIST", "Ảnh chữ số, BallTree, PCA-KNN, SVM, Bayes và lỗi dự đoán."),
        ("02", "PCA & khuôn mặt", "Digits 8×8, Olivetti, Eigenfaces, phương sai và tái tạo ảnh."),
        ("03", "Ảnh & người đi bộ", "K-Means, Spectral clustering, giảm màu và HOG + NMS."),
        ("04", "Haar-LFW", "Face/non-face gallery, Random Forest và đặc trưng Haar quan trọng."),
    ]
    for column, (number, title, text) in zip(overview, summaries):
        with column:
            st.markdown(f"<div class='eyebrow'>{number}</div><h3>{title}</h3><p>{text}</p>", unsafe_allow_html=True)
    st.info("Chạy ứng dụng bằng `streamlit run app.py`. Mở mục **Kho script** để xem nguyên văn từng file Python và tên gợi ý nếu muốn đổi.")
elif page == "MNIST":
    mnist_page()
elif page == "PCA & khuôn mặt":
    pca_page()
elif page == "Ảnh & người đi bộ":
    image_page()
elif page == "Haar-LFW":
    haar_page()
else:
    catalog_page()