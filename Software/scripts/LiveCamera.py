# scripts/LiveCamera.py

import cv2
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
    classify_image_with_rules,
)


def run_live_camera(
    camera_index: int = 0,
    thf: float = 0.003,
    ted_high: float = 0.70,
):
    """
    Start de live bankbiljet-verifier via de webcam.

    - Leest frames van de camera
    - Voert dezelfde pipeline uit als de offline dataset:
      preprocessing -> Laplacian -> Canny -> Gabor -> edge_density
      + FFT -> HF_ratio
    - Geeft een voorspelling ("Echt"/"Nep") met confidence overlay op het beeld.
    """

    print("=" * 70)
    print("LIVE CAMERA BANKBILJET-VERIFIER")
    print("Druk op 'q' of ESC om te stoppen.")
    print("=" * 70)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"FOUT: Kon camera met index {camera_index} niet openen.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("FOUT: Kon frame niet lezen van camera.")
            break

        # Optioneel: resizen voor snelheid / stabiliteit
        # frame = cv2.resize(frame, (960, 540))

        # 1) Preprocessing (zelfde als in offline pipeline)
        preprocessed = preprocess_image(frame)

        # 2) Laplacian -> Canny -> Gabor
        laplacian_img = apply_laplacian_filter(preprocessed)
        canny_img = apply_canny_edge_detection(laplacian_img)
        gabor_texture = apply_gabor_filters(canny_img)

        # Edge density op de gecombineerde edges/textuur
        edge_density = compute_edge_density(gabor_texture)

        # 3) FFT + HF-ratio
        # apply_fft verwacht een pad; daarom schrijven we het frame
        # tijdelijk naar geheugen door te encoden en opnieuw te decoden
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tmp_path = "tmp_live_frame_fft.png"
        cv2.imwrite(tmp_path, gray_frame)
        spectrum = apply_fft(tmp_path, save_path=None, visualize=False)
        hf_ratio = compute_hf_ratio(spectrum)

        # 4) Classificatie met jouw regelgebaseerde functie
        prediction, confidence = classify_rule_based(
            edge_density,
            hf_ratio,
            t_hf=thf,
            t_ed_high=ted_high,
        )

        # 5) Overlay tekst
        label_text = f"{prediction} ({confidence * 100:.1f}%)"
        info_text = f"ED={edge_density:.4f}  HF={hf_ratio:.5f}"

        color = (0, 255, 0) if prediction == "Echt" else (0, 0, 255)

        cv2.putText(
            frame,
            label_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            info_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Banknote Verifier - Live", frame)
        cv2.imshow("Edges / Texture", gabor_texture)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # q of ESC
            break

    cap.release()
    cv2.destroyAllWindows()
