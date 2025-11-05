# main.py
import os
import cv2
from Preprocessing import preprocess_image
from CannyEdge import apply_canny_edge_detection

# Padconfiguratie
BASE_INPUT_DIR = r"C:\Users\12302093\OneDrive - PXL\SH_25_26\ImageProcessing\3EAI-IP-2526-BankNote\Software\Testing\CannyEdge\Images\Input"
BASE_OUTPUT_DIR = r"C:\Users\12302093\OneDrive - PXL\SH_25_26\ImageProcessing\3EAI-IP-2526-BankNote\Software\Testing\CannyEdge\Images\Output"

# Submappen (real / fake)
subfolders = ["real", "fake"]

# Verwerk elke categorie
for subfolder in subfolders:
    input_dir = os.path.join(BASE_INPUT_DIR, subfolder)
    output_dir = os.path.join(BASE_OUTPUT_DIR, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Verwerken van map: {subfolder.upper()} ===")

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        input_path = os.path.join(input_dir, filename)
        image = cv2.imread(input_path)

        if image is None:
            print(f"[FOUT] Kan beeld niet laden: {input_path}")
            continue

        # --- Stap 1: Pre-processing ---
        preprocessed = preprocess_image(image)

        # --- Stap 2: Canny edge detection ---
        edges = apply_canny_edge_detection(preprocessed)

        # --- Stap 3: Opslaan van enkel Canny-output ---
        output_filename = os.path.splitext(filename)[0] + "_edges.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, edges)

        print(f"[OK] {filename} → {output_path}")

print("\nCanny edge detection pipeline voltooid.")
