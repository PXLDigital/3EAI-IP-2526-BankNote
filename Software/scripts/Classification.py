import cv2
import numpy as np
import os

from scripts.Preprocessing import preprocess_image
from scripts.Edges import (
    apply_laplacian_filter,
    apply_canny_edge_detection,
    compute_edge_density,
    apply_gabor_filters,
)
from scripts.FFT_analysis import (
    apply_fft,
    compute_hf_ratio,
    compute_peak_count
)


def extract_feature_vector(image_path):
    """
    Combineert edges, textuur (Gabor) en frequentiekenmerken (FFT)
    in één enkele feature vector voor classificatie.
    """

    # --- 1. Lees beeld ---
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Kan beeld niet openen: {image_path}")

    # --- 2. Preprocessing ---
    preprocessed = preprocess_image(image)

    # --- 3. Edge features (Laplacian + Canny) ---
    lap = apply_laplacian_filter(preprocessed)
    edges = apply_canny_edge_detection(lap)
    edge_density = compute_edge_density(edges)

    # Sla eenvoudige edge features op
    edge_features = np.array([edge_density], dtype=np.float32)

    # --- 4. Texture features via Gabor filters ---
    gabor_map = apply_gabor_filters(preprocessed)

    # statistische waarden van textuurkaart
    gabor_mean = np.mean(gabor_map)
    gabor_std = np.std(gabor_map)
    gabor_max = np.max(gabor_map)

    texture_features = np.array([gabor_mean, gabor_std, gabor_max], dtype=np.float32)

    # --- 5. Frequentiekenmerken via FFT ---
    magnitude_spectrum = apply_fft(image_path)

    hf_ratio = compute_hf_ratio(magnitude_spectrum)
    peak_count = compute_peak_count(magnitude_spectrum)

    fft_features = np.array([hf_ratio, peak_count], dtype=np.float32)

    # --- 6. Combineer alles in één vector ---
    feature_vector = np.concatenate([
        edge_features,       # 1 waarde
        texture_features,    # 3 waarden
        fft_features         # 2 waarden
    ])

    return feature_vector


def extract_features_from_folder(folder_path, label):
    """
    Verwerkt alle afbeeldingen in een map en geeft
    een lijst van feature vectors + labels terug.
    """
    X = []
    y = []

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(("png", "jpg", "jpeg")):
            continue

        fpath = os.path.join(folder_path, fname)
        fv = extract_feature_vector(fpath)
        X.append(fv)
        y.append(label)

    return np.array(X), np.array(y)
