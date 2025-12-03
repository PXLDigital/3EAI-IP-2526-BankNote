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
        edge_features,      # 1 waarde
        texture_features,   # 3 waarden
        fft_features        # 2 waarden
    ])

    return feature_vector, edge_density, hf_ratio


def extract_features_from_folder(folder_path, label):
    """
    Verwerkt alle afbeeldingen in een map en geeft
    een lijst van feature vectors + labels terug.
    """
    X = []
    y = []
    edge_densities = []
    hf_ratios = []

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(("png", "jpg", "jpeg")):
            continue

        fpath = os.path.join(folder_path, fname)
        try:
            fv, ed, hf = extract_feature_vector(fpath)
            X.append(fv)
            y.append(label)
            edge_densities.append(ed)
            hf_ratios.append(hf)
        except Exception as e:
            print(f"[WAARSCHUWING] Kan features niet extraheren uit {fname}: {e}")
            continue

    return np.array(X), np.array(y), edge_densities, hf_ratios


def classify_rule_based(edge_density, hf_ratio,
                        t_hf=0.003,
                        t_ed_high=0.70):
    # basis op HF
    if hf_ratio <= t_hf:
        prediction = "Echt"
    else:
        prediction = "Nep"

    # correctie met edge density voor twijfelgebied
    # twijfel: dicht bij HF-threshold
    if abs(hf_ratio - t_hf) < 0.001:
        if edge_density > t_ed_high:
            prediction = "Nep"

    # simpele confidence: afstand tot HF-threshold, met bonus als ED-consistent is
    base_conf = min(abs(hf_ratio - t_hf) / 0.003, 1.0)
    if (prediction == "Echt" and edge_density < t_ed_high) or \
       (prediction == "Nep" and edge_density >= t_ed_high):
        confidence = min(base_conf + 0.2, 1.0)
    else:
        confidence = base_conf

    return prediction, confidence


def classify_image_with_rules(image_path, t1=0.55, t2=0.01):
    """
    Classificeert een enkele afbeelding met regelgebaseerde methode.

    Parameters:
    -----------
    image_path : str
        Pad naar de afbeelding
    t1 : float
        Threshold voor edge_density
    t2 : float
        Threshold voor hf_ratio

    Returns:
    --------
    prediction : str
        "Echt" of "Nep"
    edge_density : float
        Berekende edge density
    hf_ratio : float
        Berekende HF ratio
    confidence : float
        Betrouwbaarheid van voorspelling
    """
    try:
        # Extract features
        _, edge_density, hf_ratio = extract_feature_vector(image_path)
        
        # Classify
        prediction, confidence = classify_rule_based(edge_density, hf_ratio, t1, t2)
        
        return prediction, edge_density, hf_ratio, confidence
    except Exception as e:
        print(f"[FOUT] Kan afbeelding niet classificeren: {image_path}")
        print(f"       Details: {e}")
        return None, None, None, None