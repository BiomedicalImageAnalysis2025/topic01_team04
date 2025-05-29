import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os





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




def save_image(img, name, ext="png"):
    """
    Speichert ein NumPy-Bildarray `img` in den Downloads-Ordner.
    
    - `name`: Basis-Name der Datei (ohne Extension)
    - `ext`: gewünschte Dateiendung, z.B. "png", "jpg", "tif" (default: "png")
    
    Matplotlib/Pillow erkennt aus der Extension automatisch das Format.
    """
    # 1) Pfad zum Download-Ordner (Windows, macOS, Linux)
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    # 2) kompletten Dateinamen zusammenbauen
    filename = f"{name}.{ext.lstrip('.')}"
    output_path = os.path.join(downloads, filename)
    # 3) sicherstellen, dass es den Ordner gibt
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 4) speichern
    plt.imsave(output_path, img)
    print(f"✅ Image saved to: {output_path}")





def display_images(original, name="Image"):
    """Zeige Originalbild in Jupyter an"""

    plt.figure(figsize=(12, 12))
    plt.imshow(original)
    plt.title(name)
    plt.axis('off')
    plt.show()




