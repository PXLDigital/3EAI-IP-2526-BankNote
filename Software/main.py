import cv2

from scripts.LiveCamera import run_live_camera

THRESHOLD_HF_RATIO = 0.003
THRESHOLD_EDGE_DENSITY_HIGH = 0.70

def main():
    """
    Start direct de live camera bankbiljet-verifier.
    """
    run_live_camera(
        camera_index=0,
        thf=THRESHOLD_HF_RATIO,
        ted_high=THRESHOLD_EDGE_DENSITY_HIGH,
    )

if __name__ == "__main__":
    main()
