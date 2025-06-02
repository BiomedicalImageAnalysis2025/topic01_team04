# Otsu thresholding

import cv2
import numpy as np
def otsu_thresholding(grayscale_image):
    """
    Apply Otsu's thresholding method to an grayscale image.
    
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


def plot_histogram(grayscale_image):
    """
    Plot the histogram of the grayscale image.
    
    Parameters:
    image (numpy.ndarray): Input grayscale image.
    """
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # Plot histogram
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show() 


def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale.
    
    Parameters:
    image (numpy.ndarray): Input RGB image.
    
    Returns:
    numpy.ndarray: Grayscale image.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image.")
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
