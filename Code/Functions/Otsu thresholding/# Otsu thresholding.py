# Otsu thresholding

import cv2
import numpy as np
def otsu_thresholding(grayscale_image):
    """
    Apply Otsu's thresholding method to an image.
    
    Parameters:
    image (numpy.ndarray): Input grayscale image.
    
    Returns:
    numpy.ndarray: Thresholded binary image.
    """
    # Ensure the image is in grayscale
    if len(grayscale_image.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")
    
    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_image


image_path = "/Users/daviddulkies/Documents/GitHub/topic01_team04/Code/Original_Images/Otsu/Data/NIH3T3/img/dna-0.png" 

# Bild einlesen in Graustufen
grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Otsu anwenden
binary_result = otsu_thresholding(grayscale_image)

# Ergebnis anzeigen
cv2.imshow("Otsu Thresholded", binary_result)
cv2.waitKey(0)
cv2.destroyAllWindows()