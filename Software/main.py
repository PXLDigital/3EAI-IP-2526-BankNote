import os
import cv2
import csv
import numpy as np

from scripts.Preprocessing import preprocess_image, save_image
from scripts.Edges import (
    apply_canny_edge_detection,
    apply_laplacian_filter,
    compute_edge_density,
    apply_gabor_filters,
)
from scripts.FFT_analysis import apply_fft, compute_hf_ratio, compute_peak_count
from scripts.Classification import (
    extract_features_from_folder,
    classify_rule_based,
    classify_image_with_rules
)

THRESHOLD_HF_RATIO = 0.003
# THRESHOLD_EDGE_DENSITY kun je voorlopig negeren of later gebruiken


# --- Base project directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Software/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # 3EAI-IP-2526-BankNote

# --- Input directories ---
input_dirs = [
    os.path.join(PROJECT_ROOT, "Images", "real"),
    os.path.join(PROJECT_ROOT, "Images", "fake")
]

# --- Output directories ---
preprocessed_base_dir = os.path.join(PROJECT_ROOT, "Output", "Preprocessed")
edges_base_dir = os.path.join(PROJECT_ROOT, "Output", "Edges")
fft_base_dir = os.path.join(PROJECT_ROOT, "Output", "FFT")

edge_density_output_file = os.path.join(edges_base_dir, "edge_density.csv")
fft_features_file = os.path.join(fft_base_dir, "fft_features.csv")
classification_output_file = os.path.join(PROJECT_ROOT, "Output", "classification_results.csv")

# Zorg dat hoofdoutputmappen bestaan
os.makedirs(edges_base_dir, exist_ok=True)
os.makedirs(fft_base_dir, exist_ok=True)
os.makedirs(os.path.dirname(classification_output_file), exist_ok=True)

# Verzamel edge densities, FFT features, en classificaties
all_edge_densities = {}
all_fft_features = {}
all_classifications = {}

# --- THRESHOLDS VOOR REGELGEBASEERDE CLASSIFICATIE ---
# Je kunt deze aanpassen op basis van je trainingsdata
THRESHOLD_HF_RATIO = 0.003
THRESHOLD_EDGE_DENSITY_HIGH = 0.70


print("=" * 70)
print("BANKBILJET CLASSIFICATIE PIPELINE - MET REGELGEBASEERDE CLASSIFIER")

# --- Loop over elke datasetmap (real / fake) ---
for input_dir in input_dirs:
    label = os.path.basename(os.path.normpath(input_dir))  # 'real' of 'fake'
    preprocessed_output_dir = os.path.join(preprocessed_base_dir, label)
    edges_output_dir = os.path.join(edges_base_dir, label)
    fft_output_dir = os.path.join(fft_base_dir, label)
    
    os.makedirs(preprocessed_output_dir, exist_ok=True)
    os.makedirs(edges_output_dir, exist_ok=True)
    os.makedirs(fft_output_dir, exist_ok=True)

    print(f"\n=== Verwerken van map: {label.upper()} ===")
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        input_path = os.path.join(input_dir, filename)
        image = cv2.imread(input_path)

        if image is None:
            print(f"[FOUT] Kan beeld niet laden: {input_path}")
            continue

        try:
            # --- Stap 1: Preprocessing ---
            preprocessed = preprocess_image(image)
            preprocessed_filename = os.path.splitext(filename)[0] + "_preprocessed.png"
            save_image(preprocessed, preprocessed_output_dir, preprocessed_filename)

            # --- Stap 2: Laplacian + Canny gecombineerd ---
            laplacian_image = apply_laplacian_filter(preprocessed)
            canny_image = apply_canny_edge_detection(laplacian_image)
            combined_edges = apply_gabor_filters(canny_image)
            combined_filename = os.path.splitext(filename)[0] + "_edges_combined.png"
            save_image(combined_edges, edges_output_dir, combined_filename)

            # --- Stap 3: Edge density berekenen ---
            density = compute_edge_density(combined_edges)
            all_edge_densities[f"{label}/{filename}"] = density

            # --- Stap 4: FFT-analyse ---
            fft_filename = os.path.splitext(filename)[0] + "_FFT_spectrum.png"
            fft_save_path = os.path.join(fft_output_dir, fft_filename)

            spectrum = apply_fft(input_path, save_path=fft_save_path, visualize=False)
            hf_ratio = compute_hf_ratio(spectrum)
            peak_count = compute_peak_count(spectrum)
            all_fft_features[f"{label}/{filename}"] = (hf_ratio, peak_count)

            # --- Stap 5: REGELGEBASEERDE CLASSIFICATIE ---
            prediction, confidence = classify_rule_based(
                density,
                hf_ratio,
                t_hf=THRESHOLD_HF_RATIO,
                t_ed_high=THRESHOLD_EDGE_DENSITY_HIGH
)

            
            all_classifications[f"{label}/{filename}"] = {
                'prediction': prediction,
                'confidence': confidence,
                'edge_density': density,
                'hf_ratio': hf_ratio,
                'true_label': label
            }

            print(f"[OK] {filename}")
            print(f"     → Edge Density: {density:.4f} | HF Ratio: {hf_ratio:.4f}")
            print(f"     → Prediction: {prediction} | Confidence: {confidence:.2%}")

        except Exception as e:
            print(f"[FOUT] {filename}: {e}")
            continue

# --- Stap 6: Edge densities naar CSV ---
print("\n" + "=" * 70)
print("OPSLAAN VAN RESULTATEN")
print("=" * 70)

with open(edge_density_output_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Filename", "EdgeDensity"])
    for filename, density in all_edge_densities.items():
        writer.writerow([filename, density])

print(f"✓ Edge densities opgeslagen in: {edge_density_output_file}")

# --- Stap 7: FFT-features naar CSV ---
with open(fft_features_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Filename", "HF_ratio", "PeakCount"])
    for filename, (hf_ratio, peak_count) in all_fft_features.items():
        writer.writerow([filename, hf_ratio, peak_count])

print(f"✓ FFT-features opgeslagen in: {fft_features_file}")

# --- Stap 8: CLASSIFICATIE RESULTATEN naar CSV ---
with open(classification_output_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "Filename", 
        "True_Label", 
        "Prediction", 
        "Correct", 
        "Confidence",
        "Edge_Density",
        "HF_Ratio"
    ])
    
    correct_predictions = 0
    total_predictions = 0
    
    for filename, result in all_classifications.items():
        is_correct = (result['true_label'] == 'real' and result['prediction'] == 'Echt') or \
                     (result['true_label'] == 'fake' and result['prediction'] == 'Nep')
        
        correct_value = 1 if is_correct else 0

        writer.writerow([
            filename,
            result['true_label'],
            result['prediction'],
            correct_value,
            f"{result['confidence']:.4f}",
            f"{result['edge_density']:.4f}",
            f"{result['hf_ratio']:.6f}"
        ])
        
        if is_correct:
            correct_predictions += 1
        total_predictions += 1

print(f"✓ Classificatie resultaten opgeslagen in: {classification_output_file}")

# --- Stap 9: Statistieken ---
if total_predictions > 0:
    accuracy = correct_predictions / total_predictions
    print(f"\n" + "=" * 70)
    print(f"CLASSIFICATIE STATISTIEKEN")
    print(f"=" * 70)
    print(f"Totale voorspellingen: {total_predictions}")
    print(f"Correcte voorspellingen: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
else:
    print("[WAARSCHUWING] Geen voorspellingen gemaakt")