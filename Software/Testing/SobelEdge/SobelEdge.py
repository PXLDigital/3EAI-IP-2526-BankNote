import cv2
import numpy as np

def apply_sobel_edge(image):
    """
    Voer Sobel edge detection uit op een (kleur of grijs) beeld.
    Parameters:
        image (numpy.ndarray): Ingangsbeeld (BGR of grijswaarden)
    Returns:
        sobel_result (numpy.ndarray): Beeld met Sobel edge-detectie
    """
    # Converteer naar grijswaarden als het een kleurbeeld is
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Bereken Sobel gradiënten
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Combineer X en Y tot magnitude
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    sobel_magnitude = sobel_magnitude.astype(np.uint8)

    return sobel_magnitude
