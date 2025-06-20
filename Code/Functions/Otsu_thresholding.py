# Otsu thresholding

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


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
    hist = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])
    
    # Plot histogram
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


def display_images(original, name="Image"):
    """show original image"""

    plt.figure(figsize=(4, 4))
    plt.imshow(original, cmap='gray', vmin=0, vmax=255) 
    plt.title(name)
    plt.axis('off')
    plt.show()


def save_image(img, name, ext="tiff"):
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    filename = f"{name}.{ext.lstrip('.')}"
    output_path = os.path.join(downloads, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success = cv2.imwrite(output_path, img)
    if success:
        print(f"Bild gespeichert unter: {output_path}")
    else:
        print(f"❌ Fehler beim Speichern von: {output_path}")




