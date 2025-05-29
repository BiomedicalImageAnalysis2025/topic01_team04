import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gaussian_filter(input_path, kernel_size=5):
    """
    Wende Gauß-Filter auf ein Bild an und zeige es in Jupyter an
    
    Parameter:
    input_path (str): Pfad zum Eingangsbild
    kernel_size (int): Größe des Filterkerns (muss ungerade sein)
    """
    # Bild im RGB-Format laden
    image = cv2.imread(input_path)
    
    if image is None:
        print(f"Fehler: Bild konnte nicht geladen werden: {input_path}")
        return None
    
    # Konvertierung von BGR (OpenCV) zu RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Gauß-Filter anwenden
    if kernel_size % 2 == 0:  # Sicherstellen, dass Kernel ungerade ist
        kernel_size += 1
    blurred = cv2.GaussianBlur(image_rgb, (kernel_size, kernel_size), 0)
    
    return image_rgb, blurred




def display_image(original, filtered):
    """Zeige Original und gefiltertes Bild nebeneinander"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original Bild')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(filtered)
    plt.title('Gauß-Gefiltertes Bild')
    plt.axis('off')
    
    plt.show()
