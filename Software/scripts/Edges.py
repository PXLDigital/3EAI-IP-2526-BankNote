import cv2
import numpy as np

def apply_canny_edge_detection(
    image,
    sigma=0.33,
    use_bilateral=True,
    morph_cleanup=True,
    enhance_edges=True,
    debug=False
):
    """
    Geavanceerde Canny edge detection voor bankbiljetanalyse.
    Werkt op een gepreprocessed BGR-beeld (na HSV + CLAHE + blur).

    Verbeteringen t.o.v. standaardversie:
      - Adaptieve drempels op basis van histogram i.p.v. enkel mediaan.
      - Combinatie van bilateral + Gaussian filtering voor balans tussen detail en ruis.
      - Optionele randversterking (morfologische dilatie).
      - Meer robuuste resultaten tussen echte en valse biljetten.

    Parameters:
        image (np.ndarray): Inputbeeld (BGR).
        sigma (float): Parameter voor automatische thresholdberekening (0.33 = standaard).
        use_bilateral (bool): True = gebruik bilateral filter voor detailbehoud.
        morph_cleanup (bool): True = verwijder kleine ruis via morfologische filters.
        enhance_edges (bool): True = versterk dunne randen lichtjes.
        debug (bool): True = toon tussentijdse beelden (voor analyse).

    Returns:
        edges (np.ndarray): Binair beeld (wit = rand, zwart = geen rand).
    """

    # --- 1. Controleer of het beeld kleur of grijs is ---
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()  # beeld is al grijs

    # --- 2. Filtering: ruisonderdrukking met behoud van textuur ---
    if use_bilateral:
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=40)
        gray = cv2.GaussianBlur(gray, (3, 3), 0.5)
    else:
        gray = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # --- 3. Adaptieve thresholds berekenen op basis van histogram ---
    v = np.median(gray)
    # Gebruik percentielen voor robuustere drempels
    lower = int(max(0, np.percentile(gray, 25)))
    upper = int(min(255, np.percentile(gray, 75)))
    # Corrigeer met sigma
    lower = int(max(0, lower * (1.0 - sigma)))
    upper = int(min(255, upper * (1.0 + sigma)))

    # --- 4. Canny edge detection uitvoeren ---
    edges = cv2.Canny(gray, lower, upper)

    # --- 5. Morfologische filtering ---
    if morph_cleanup:
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    # --- 6. Optionele randversterking ---
    if enhance_edges:
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    # --- 7. Debug visualisatie ---
    if debug:
        cv2.imshow("Grijsbeeld", gray)
        cv2.imshow("Edges (Canny)", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return edges


def compute_edge_density(edges):
    """
    Bereken de edge density (percentage randpixels t.o.v. totaal).
    Parameters:
        edges (np.ndarray): Binair randbeeld.
    Returns:
        float: Edge density (tussen 0 en 1).
    """
    total_pixels = edges.size
    edge_pixels = np.count_nonzero(edges)
    return edge_pixels / total_pixels


def apply_laplacian_filter(image, kernel_size=3, scale=1, delta=0, debug=False):
    """
    Toepassing van Laplacian-filter om hoge frequenties te versterken.
    Accentueert fijne texturen en microstructuren in bankbiljetten.

    Parameters:
        image (np.ndarray): Inputbeeld (BGR of grijs).
        kernel_size (int): Kernelgrootte (meestal 1, 3 of 5).
        scale (float): Schaalfactor voor versterking.
        delta (float): Offset toegevoegd aan resultaat.
        debug (bool): Toon resultaten visueel (optioneel).

    Returns:
        laplacian_enhanced (np.ndarray): Versterkt grijsbeeld.
    """
    # --- 1. Controleer of het beeld kleur of grijs is ---
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()  # beeld is al grijs

    # --- 2. Bereken Laplacian ---
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size, scale=scale, delta=delta)
    lap_abs = cv2.convertScaleAbs(lap)

    # --- 3. Combineer met origineel (accentuatie) ---
    laplacian_enhanced = cv2.addWeighted(gray, 0.7, lap_abs, 0.7, 0)

    # --- 4. Optioneel: debug tonen ---
    if debug:
        cv2.imshow("Laplacian Enhanced", laplacian_enhanced)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return laplacian_enhanced

def apply_gabor_filters(image, frequencies=[0.1, 0.2, 0.3], orientations=8, debug=False):
    """
    Toepassing van Gabor-filters om oriëntatie- en frequentiespecifieke texturen te detecteren.
    Parameters:
        image (np.ndarray): Inputbeeld (BGR of grijs)
        frequencies (list): Ruimtelijke frequenties (1 / golflengte)
        orientations (int): Aantal richtingen (bijv. 8 = 0°, 22.5°, ... 157.5°)
    Returns:
        texture_energy (np.ndarray): Gecombineerde textuursterktekaart
    """
    # --- 1. Converteer naar grijs ---
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    rows, cols = gray.shape
    accum = np.zeros_like(gray, dtype=np.float32)

    # --- 2. Loop over oriëntaties en frequenties ---
    for theta in np.linspace(0, np.pi, orientations, endpoint=False):
        for freq in frequencies:
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),  # grootte van filtervenster
                sigma=4.0,       # breedte van de Gaussiaan
                theta=theta,     # oriëntatie
                lambd=1/freq,    # golflengte
                gamma=0.5,       # anisotropie: 0.5 = elliptisch
                psi=0,           # faseverschuiving
                ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            accum = np.maximum(accum, filtered)  # neem maximumrespons

    # --- 3. Normaliseer ---
    texture_energy = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if debug:
        cv2.imshow("Gabor Texture Map", texture_energy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return texture_energy

