import os
import cv2
from scripts.Preprocessing import preprocess_image, save_image

# Input en output directories
input_dirs = ['images/real/', 'images/fake/']
output_base_dir = 'output/preprocessed/'

# Doorloop elke datasetmap (real / fake)
for input_dir in input_dirs:
    label = os.path.basename(os.path.normpath(input_dir))  # 'real' of 'fake'
    output_dir = os.path.join(output_base_dir, label)
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            image = cv2.imread(input_path)
            
            if image is None:
                print(f"[FOUT] Kan beeld niet laden: {input_path}")
                continue
            
            # Pre-process het beeld
            processed_image = preprocess_image(image)
            
            # Sla het resultaat op
            output_filename = filename.replace('.png', '_processed.png').replace('.jpg', '_processed.jpg')
            save_image(processed_image, output_dir, output_filename)
            
            print(f"[OK] {input_path} → {os.path.join(output_dir, output_filename)}")

print("Pre-processing voltooid.")
