import os
import cv2
import csv
from scripts.Preprocessing import preprocess_image, save_image
from scripts.Edges import (
    apply_canny_edge_detection,
    apply_laplacian_filter,
    compute_edge_density,
    apply_gabor_filters,
)
from scripts.FFT_analysis import apply_fft, compute_hf_ratio, compute_peak_count

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

# Zorg dat hoofdoutputmappen bestaan
os.makedirs(edges_base_dir, exist_ok=True)
os.makedirs(fft_base_dir, exist_ok=True)

# Verzamel edge densities
all_edge_densities = {}
all_fft_features = {}

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

        print(f"[OK] {filename} → "
              f"Edges: {combined_filename}, Edge density: {density:.4f}")

        # --- Stap 4: FFT-analyse ---
        fft_filename = os.path.splitext(filename)[0] + "_FFT_spectrum.png"
        fft_save_path = os.path.join(fft_output_dir, fft_filename)

        try:
            spectrum = apply_fft(input_path, save_path=fft_save_path, visualize=False)
            hf_ratio = compute_hf_ratio(spectrum)
            peak_count = compute_peak_count(spectrum)
            all_fft_features[f"{label}/{filename}"] = (hf_ratio, peak_count)
            print(f"[FFT] {filename} → Spectrum opgeslagen, HF_ratio: {hf_ratio:.4f}, Peak count: {peak_count}")
        except Exception as e:
            print(f"[FOUT bij FFT] {filename}: {e}")

# --- Stap 5: Edge densities naar CSV ---
with open(edge_density_output_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Filename", "EdgeDensity"])
    for filename, density in all_edge_densities.items():
        writer.writerow([filename, density])

# --- Stap 6: FFT-features naar CSV ---
with open(fft_features_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Filename", "HF_ratio", "PeakCount"])
    for filename, (hf_ratio, peak_count) in all_fft_features.items():
        writer.writerow([filename, hf_ratio, peak_count])

print(f"\nPipeline voltooid!")
print(f"- Gecombineerde edge densities opgeslagen in: {edge_density_output_file}")
print(f"- FFT-spectra opgeslagen in: {fft_base_dir}")
print(f"- FFT-features (HF_ratio + PeakCount) opgeslagen in: {fft_features_file}")
