import os
import cv2
import numpy as np
from Preprocessing import preprocess_image, save_image

# Bepaal het pad van dit script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input- en outputmappen
input_dirs = ['images/real/', 'images/fake/']
preproc_output_base = os.path.join(BASE_DIR, 'Output')
sobel_output_base = os.path.join(BASE_DIR, 'Output', 'Sobeledge')

# Maak de hoofdmap voor Sobel-output aan
os.makedirs(sobel_output_base, exist_ok=True)

# Doorloop elke datasetmap (real / fake)
for input_dir in input_dirs:
    label = os.path.basename(os.path.normpath(input_dir))  # 'real' of 'fake'

    # Outputmappen aanmaken
    preproc_output_dir = os.path.join(preproc_output_base, label)
    sobel_output_dir = os.path.join(sobel_output_base, label)
    os.makedirs(preproc_output_dir, exist_ok=True)
    os.makedirs(sobel_output_dir, exist_ok=True)

    # Verwerking per afbeelding
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            image = cv2.imread(input_path)

            if image is None:
                print(f"[FOUT] Kan beeld niet laden: {input_path}")
                continue

            # --- STAP 1: Pre-processing ---
            processed_image = preprocess_image(image)

            # Sla preprocessed beeld op
            base_name, ext = os.path.splitext(filename)
            processed_filename = f"{base_name}_processed{ext}"
            processed_path = os.path.join(preproc_output_dir, processed_filename)
            save_image(processed_image, preproc_output_dir, processed_filename)
            print(f"[OK] Preprocessed: {input_path} → {processed_path}")

            # --- STAP 2: Sobel edge detection ---
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
            sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            sobel_magnitude = sobel_magnitude.astype(np.uint8)

            # Sla Sobel-resultaat op in overeenkomstige real/fake map
            sobel_filename = f"{base_name}_processed_sobel.png"
            sobel_path = os.path.join(sobel_output_dir, sobel_filename)
            cv2.imwrite(sobel_path, sobel_magnitude)
            print(f"[OK] Sobel Edge: {processed_path} → {sobel_path}")

print("\n✅ Alle afbeeldingen zijn succesvol gepreprocessed én met Sobel gefilterd!")
print(f"Sobel resultaten opgeslagen in: {sobel_output_base}")
