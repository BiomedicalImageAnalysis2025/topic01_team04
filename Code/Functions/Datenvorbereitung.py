import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.pyplot as plt
from skimage import color, segmentation, filters, morphology, io
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import imageio.v3 as iio




def apply_gaussian_filter(input_path, kernel_size=5):
    """
    Wendet einen Gauß-Filter auf ein Bild und gibt es zurück
    
    Parameter:
    input_path (str): Pfad zum Eingangsbild
    kernel_size (int): Größe des Kernels (muss ungerade sein)
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
    
    return  blurred





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
    return filtered





def save_image(img, name, ext="png"):
    """
    Speichert ein NumPy-Bildarray `img` in den Downloads-Ordner.
    
    - `name`: Basis-Name der Datei 
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
    # 4) speichern
    if img.ndim == 2:
        plt.imsave(output_path, img)
    else:
        plt.imsave(output_path, img)

    
def save_image_grey(img, name, ext="png"):
    """
    Speichert ein Graustufenbild (2D) im uint8-Format ohne zusätzliche Normalisierung.
    """
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    filename = f"{name}.{ext.lstrip('.')}"
    output_path = os.path.join(downloads, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, img)
    print(f"Image saved to: {output_path}")
    



def display_images(original, name="Image"):
    """Zeige Originalbild an"""

    plt.figure(figsize=(4, 4))
    plt.imshow(original)
    plt.title(name)
    plt.axis('off')
    plt.show()

def display_images_grey(original, name="Image"):
    """Zeige Originalbild an"""

    plt.figure(figsize=(4, 4))
    plt.imshow(original, cmap='gray')
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





def norm_255(image: np.ndarray) -> np.ndarray:
    """
    Normalisiert ein RGB-Bild (NumPy-Array) auf den Wertebereich [0,1].
    Liefert ein neues float32-Array zurück.
    """
    # Eingabevalidierung: muss NumPy-Array mit 3 Dimensionen und 3 Farbkanälen sein
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image expected with shape (H, W, 3)")
    
    # Neue Kopie im float32-Format erstellen
    normalized_img = image.astype(np.float32) # Konvertierung in float32
    
    # Werte [0,255] auf [0,1] skalieren 
    normalized_img /= 255.0 # Normalisierung auf [0,1]
    
    # Optional falls vorhanden: Werte außerhalb [0,1] abschneiden
    normalized_img = np.clip(normalized_img, 0.0, 1.0) 
    
    return normalized_img



     
def z_normalize(rgb_image: np.ndarray) -> np.ndarray:
    """
    Führt eine Z-Transformation (Standardisierung) auf einem RGB-Bild durch,
    wobei Kanäle mit σ=0 nicht verändert werden (bleiben 0).
    """
    # Eingabeprüfung:
    if not isinstance(rgb_image, np.ndarray):  
        raise TypeError("Eingabebild muss ein NumPy-Array sein.")
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Eingabebild muss die Form (H, W, 3) haben (RGB).")

    # Bild in float32 konvertieren 
    img_float = rgb_image.astype(np.float32, copy=True)

    # Pixelmatrix (H*W, 3)
    flat = img_float.reshape(-1, 3) 

    # Mittelwert und Standardabweichung berechnen
    means = flat.mean(axis=0) # Mittelwert für jeden Kanal
    stds  = flat.std(axis=0, ddof=0) # Standardabweichung für jeden Kanal
    # "Schutz" gegen Division durch 0
    # Hier ersetzen wir std == 0 durch 1 → verhindert division durch 0
    safe_stds = np.where(stds == 0, 1.0, stds)  

    # Umformen für Broadcasting
    means_reshaped = means.reshape((1, 1, 3)) 
    stds_reshaped  = safe_stds.reshape((1, 1, 3))  # jetzt safe_stds statt stds

    # ✅ Z-Transformation mit geschützten Standardabweichungen
    z_image = (img_float - means_reshaped) / stds_reshaped  # Division durch safe_stds

    return z_image






def save_as_numpy(array: np.ndarray, name: str):
    """
    Speichert ein beliebiges NumPy-Array im Downloads-Ordner.
    Der Dateiname wird aus dem übergebenen `name` gebildet (mit .npy-Endung).
    Beispiel-Aufruf im Main-Code:

    Parameter:
    -----------
    array : np.ndarray
        Das NumPy-Array, das gespeichert werden soll (z. B. ein z-transformiertes Bild).
    name : str
        Basis-Name der Ausgabedatei (ohne Endung). Die Funktion fügt automatisch ".npy" hinzu.

    Rückgabe:
    ---------
    output_path : str
        Der komplette Pfad der gespeicherten .npy-Datei im Downloads-Ordner.
    """
    # 1) Pfad zum aktuellen Downloads-Ordner (cross-platform)
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    
    # 2) Stelle sicher, dass der Downloads-Ordner existiert (sollte normalerweise der Fall sein)
    os.makedirs(downloads_folder, exist_ok=True)
    
    # 3) Dateiname mit .npy-Endung
    filename = name if name.lower().endswith(".npy") else f"{name}.npy"
    
    # 4) Vollständiger Pfad zur Ausgabedatei
    output_path = os.path.join(downloads_folder, filename)
    
    # 5) Array verlustfrei als .npy speichern
    np.save(output_path, array)
    
    print(f"Array erfolgreich gespeichert: {output_path}")
    return output_path





def apply_gaussian_to_array(image_array, kernel_size=5):
    """
    Wendet einen Gauß-Filter auf einen Array an.

    Rückgabe: uint8-Array (0–255)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Umwandlung in float32 für Filter
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)

    blurred = cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)

    # Rückwandlung in uint8 für einheitliche Weiterverarbeitung
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    return blurred




def apply_bilateral_to_array(image_array, kernel_size=5, sigma_color=75, sigma_space=75):
    """
    Wendet einen bilateralen Filter auf ein Array an.

    Rückgabe: uint8-Array (0–255)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Bilateral braucht uint8
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    filtered = cv2.bilateralFilter(
        src=image_array,
        d=kernel_size,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )
    return filtered



def apply_median_filter(image, kernel=3):
    """
    Wendet einen Medianfilter auf das gegebene Bild an.

    Parameter:
    - image: np.ndarray
    - kernel: int, die Größe des Median-Kernels (muss ungerade sein)

    """
    if kernel % 2 == 0 or kernel < 1:
        raise ValueError("Kernelgröße muss eine ungerade positive Zahl sein (z.B. 3, 5, 7).")
    
    filtered = cv2.medianBlur(image, kernel)
    return filtered



def apply_gauß_to_string(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Wendet einen Gauß-Filter mit gegebener Kernel-Größe auf ein Bild an.
    

    Rückgabe:
    - Gefiltertes Bild als NumPy-Array
    """
    if kernel_size % 2 == 0:


        raise ValueError("Kernel-Größe muss ungerade sein (z. B. 3, 5, 7).")

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)




def apply_bilateral_to_string(image: np.ndarray, kernel_size: int, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Wendet einen bilateralen Filter mit gegebener Kernel-Größe auf ein Bild an.

    Parameter:
    - image: Eingabebild als NumPy-Array (z.B. Hue-Kanal, RGB, etc.)
    - kernel_size: Durchmesser des Pixel-Nachbarschaftsbereichs (muss ungerade sein)
    - sigma_color: Filterstärke für Farbunterschiede (Standard 75)
    - sigma_space: Filterstärke für räumliche Nähe (Standard 75)

    Rückgabe:
    - Gefiltertes Bild als NumPy-Array
    """

    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("Kernel-Größe muss eine positive ungerade Zahl sein")

    # d in bilateralFilter ist der Durchmesser der Nachbarschaft
    filtered_image = cv2.bilateralFilter(image, d=kernel_size, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    return filtered_image


def remove_alpha_channel(image):
 
    if image.ndim == 3 and image.shape[2] == 4: 
        return image[:, :, :3]  # RGBA → RGB
    elif image.ndim == 3 and image.shape[2] == 3:
        return image  # RGB bleibt 
    

  
def apply_watershed(image, num_markers):
    """
    Wendet Wasserquellen basierte Watershed auf ein Bild an.

    Rückgabe:
        labels_ws (ndarray): Segmentiertes Bild
    """
    # 1. Alpha-Kanal entfernen 
    img_rgb = np.copy(image)
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
        img_rgb = img_rgb[:, :, :3]
    elif image.ndim == 3 and image.shape[2] == 3:
        return image 

    # 2. In Graustufen umwandeln
    img_gray = color.rgb2gray(img_rgb) if img_rgb.ndim == 3 else img_rgb

    # 3. Binärmaske erzeugen 
    th = filters.threshold_otsu(img_gray)
    binary = img_gray > th

    # 4. Distanz-Transform 
    distance = ndi.distance_transform_edt(binary)

    # 5. Marker setzen 
    coords = peak_local_max(
        distance, 
        labels=binary,
        num_peaks=num_markers,
        footprint=np.ones((15, 15))
    )
    markers = np.zeros(distance.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx

    #  6. Watershed anwenden
    labels_ws = segmentation.watershed(
        -distance,
        markers,
        mask=binary
    )

    # 7. Bildanzeige 
    # Erstes Plotfenster: Maske + Marker
    fig1, axes = plt.subplots(1, 2, figsize=(5, 5))
    ax1 = axes.ravel()
    ax1[0].imshow(img_gray, cmap='gray')
    ax1[0].contour(binary, [0.5], colors='r')
    ax1[0].set_title('Maske')

    ax1[1].imshow(distance, cmap='magma')
    ax1[1].plot(coords[:, 1], coords[:, 0], 'b.')
    ax1[1].set_title(f'Distanz + {num_markers} Marker')

    for a in ax1:
        a.axis('off')
    plt.tight_layout()
    plt.show()

    # Zweites Plotfenster: Segmentierte Zellen separat
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.imshow(color.label2rgb(labels_ws, image=img_gray, bg_label=0))
    ax2.set_title('Segmentierte Zellen (Watershed)')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

    return labels_ws





def apply_watershed_2(image, min_distance=60, thresh_rel=0.6):
    """
    Parameter:
    
        min_distance (int): Mindestabstand zwischen Zellkernen (Pixel).
        thresh_rel (float): Relativer Schwellwert für Markerplatzierung (0–1).

    Rückgabe:
        labels_ws : Label-Bild mit segmentierten Zellen.
    """
    # 1. Alpha-Kanal entfernen + Graustufen 
    img = np.copy(image)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    img_gray = color.rgb2gray(img) if img.ndim == 3 else img

    # 2. Binarisierung + Morphologie 
    binary = img_gray > filters.threshold_otsu(img_gray)
    binary = morphology.binary_closing(binary, morphology.disk(3))
    binary = morphology.binary_opening(binary, morphology.disk(5))

    # 3. Distanztransformation 
    distance = ndi.distance_transform_edt(binary)

    #  4. Marker setzen 
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_rel=thresh_rel,
        footprint=np.ones((5, 5)),
        labels=binary
    )
    markers = np.zeros(distance.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    #  5. Watershed 
    labels_ws = segmentation.watershed(
        -distance,
        markers,
        mask=binary,
        watershed_line=True
    )

    #  6. Visualisierung 
    # Erstes Fenster: Original + Marker und Distanzbild nebeneinander
    fig1, ax1 = plt.subplots(1, 2, figsize=(5, 5))

    ax1[0].imshow(img_gray, cmap='gray')
    ax1[0].plot(coords[:, 1], coords[:, 0], 'r.', markersize=5)
    ax1[0].set_title(f'Original mit {len(coords)} Markern')

    ax1[1].imshow(distance, cmap='viridis')
    ax1[1].set_title('Distanztransformation')

    for a in ax1:
        a.axis('off')
    plt.tight_layout()
    plt.show()

    # Zweites Fenster: Segmentiertes Ergebnis separat
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.imshow(labels_ws, cmap='nipy_spectral')
    ax2.set_title(f'Segmentierte Zellen: {len(coords)}')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

    return labels_ws





def save_raw_image(img, name, ext="tiff"):
    """
    Speichert Bild mit unveränderten Pixeldaten, z. B. für Analyse (Dice, KMeans).
    """
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    filename = f"{name}.{ext}"
    output_path = os.path.join(downloads, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    iio.imwrite(output_path, img)  # kein Clip, kein Umwandeln
    print(f"RAW image saved to: {output_path}")
