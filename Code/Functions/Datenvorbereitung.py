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


# Definition mit Defaults für sigma_color und sigma_space
def apply_bilateral_filter(input_path, kernel_size=5,
                           sigma_color=75, sigma_space=75):
    """
    Wendet einen bilateralen Filter an und liefert (original, gefiltert).
    
    Parameter:
      input_path   (str): Pfad zum Bild
      kernel_size  (int): Kernel-Durchmesser (ungerade)
      sigma_color  (int): Farb-Glättung (default: 75)
      sigma_space  (int): Raum-Glättung (default: 75)
    Rückgabe:
      Tuple (original_rgb, filtered_rgb)
    """
    image = cv2.imread(input_path)
    if image is None:
        print(f"Fehler: Bild konnte nicht geladen werden: {input_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if kernel_size % 2 == 0:
        kernel_size += 1

    filtered = cv2.bilateralFilter(
        src=image_rgb,
        d=kernel_size,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )
    return image_rgb, filtered




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


def split_channels(img_array):
     
     if img_array.ndim != 3 or img_array.shape[2] != 3:
        raise ValueError("Das Eingabebild muss die Form (H, W, 3) haben (RGB).")
     else:
    #reshape image to dimensions (hight*with of image, 3(dimension of colours R, G and B))
        reshaped_image = img_array.reshape(-1, 3)
        R = reshaped_image[:,0]
        G = reshaped_image[:,1]
        B = reshaped_image[:,2]
        return R, G, B
     
     

def remove_red_channel(rgb_image):
    """
    Entfernt den Rot-Kanal eines RGB-Bildes (setzt ihn auf 0)
    und gibt ein Bild mit nur Grün- und Blau-Anteilen zurück.

    Parameter:
    rgb_image (np.ndarray): RGB-Bild im Format (H, W, 3)

    Rückgabe:
    np.ndarray: RGB-Bild mit Rot-Kanal = 0
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Eingabebild muss die Form (H, W, 3) haben (RGB).")

    # Kopiere das Bild, um das Original nicht zu verändern
    gb_image = rgb_image.copy()
    
    # Setze Rotkanal (Index 0 in RGB) auf 0
    gb_image[..., 0] = 0
    
    return gb_image


def remove_channel(rgb_image, channel='red'):
    """
    Entfernt einen bestimmten Farbkanal eines RGB-Bildes (setzt ihn auf 0).

    Parameter:
    rgb_image (np.ndarray): RGB-Bild im Format (H, W, 3)
    channel (str): Zu entfernender Kanal ('red', 'green', 'blue')

    Rückgabe:
    np.ndarray: RGB-Bild mit einem auf 0 gesetzten Farbkanal
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Eingabebild muss die Form (H, W, 3) haben (RGB).")

    # Zuordnung: Farbname → Index
    channel_map = {'red': 0, 'green': 1, 'blue': 2}
    
    if channel not in channel_map:
        raise ValueError("Ungültiger Kanal. Erlaubt: 'red', 'green', 'blue'.")

    # Bild kopieren, um das Original nicht zu verändern
    modified_image = rgb_image.copy()
    
    # Entfernen des gewünschten Kanals
    modified_image[..., channel_map[channel]] = 0
    
    return modified_image

     
