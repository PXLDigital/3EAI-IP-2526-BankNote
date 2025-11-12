# scripts/FFT_analysis.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

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

def compute_hf_ratio(magnitude_spectrum, inner_radius_ratio=0.3, outer_radius_ratio=0.5, top_n=100):
    """
    Bereken HF_ratio: energie van top-N hoge-frequentie pieken in een ring t.o.v. totale energie.
    
    Parameters:
        magnitude_spectrum : 2D np.array
            FFT magnitude spectrum.
        inner_radius_ratio : float
            Binnenste radius van de hoge-frequentie ring (0 = centrum, 1 = max radius).
        outer_radius_ratio : float
            Buitenste radius van de hoge-frequentie ring.
        top_n : int
            Aantal hoogste magnitudes in de ring om mee te tellen.
    """
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_radius = 0.5 * np.sqrt(h**2 + w**2)  # half-diagonaal

    inner_radius = inner_radius_ratio * max_radius
    outer_radius = outer_radius_ratio * max_radius

    # Mask: alleen pixels in de hoge-frequentie ring
    ring_mask = (distance >= inner_radius) & (distance <= outer_radius)

    # Alle magnitudes in de ring
    ring_magnitudes = magnitude_spectrum[ring_mask]

    if len(ring_magnitudes) == 0:
        return 0.0

    # Top N magnitudes in de ring
    top_magnitudes = np.sort(ring_magnitudes)[-top_n:]
    hf_energy = np.sum(top_magnitudes**2)
    total_energy = np.sum(magnitude_spectrum**2)

    return hf_energy / total_energy

def compute_peak_count(magnitude_spectrum, threshold_ratio=0.1):
    """
    Tel het aantal significante pieken in het spectrum.
    """
    profile = np.mean(magnitude_spectrum, axis=0)
    threshold = threshold_ratio * np.max(profile)
    peaks, _ = find_peaks(profile, height=threshold)
    return len(peaks)
