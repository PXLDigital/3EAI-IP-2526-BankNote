# scripts/FFT_analysis.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_fft(image_path, save_path=None, visualize=False):
    """
    Voert een 2D FFT uit op een grijswaardenbeeld en toont of bewaart het frequentiespectrum.

    Parameters:
        image_path (str): pad naar invoerbeeld.
        save_path (str, optioneel): pad om het spectrum op te slaan.
        visualize (bool): toon het resultaat in een venster.
    """
    # 1. Lees beeld in grijswaarden
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Afbeelding niet gevonden: {image_path}")

    # 2. Bereken 2D FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # 3. Bereken magnitude spectrum (log-schaal)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # 4. Visualisatie
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Origineel beeld")
        plt.imshow(gray, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Magnitude Spectrum (FFT)")
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.axis('off')
        plt.show()

    # 5. Opslaan
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, np.uint8(255 * magnitude_spectrum / np.max(magnitude_spectrum)))

    return magnitude_spectrum

# Voorbeeldgebruik:
# spectrum = apply_fft("Input/real/10euroReal.png", "Output/FFT/10euroReal_spectrum.png", visualize=True)
