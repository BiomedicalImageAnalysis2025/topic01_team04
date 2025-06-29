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


"""
This python file contains all functions for image processing which were use in "Datenvorbereitung.ipynb".

All AI-written or enhanced code is clearly marked. 
Note: All `save_image` functions using "os" were partially written by AI.

Author: David Schroth
Last Change: 2025-06-27

"""






def apply_gaussian_filter(input_path, kernel_size=5):
    """
    Applies a Gaussian filter to an image

    Parameters:
    input_path (str): Path to the input image
    kernel_size (int): Size of the kernel (must be odd)
"""

    # Load Image
    image = cv2.imread(input_path)
    
    # Konvertierung von BGR (OpenCV) zu RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Gauß-Filter anwenden
    if kernel_size % 2 == 0:  # Sicherstellen, dass Kernel ungerade ist
        kernel_size += 1
    blurred = cv2.GaussianBlur(image_rgb, (kernel_size, kernel_size), 0) #hier wird der Kernel auf 0 gesetzt, damit er automatisch berechnet wird
    
    return  blurred




def apply_bilateral_filter(input_path, kernel_size=5,
                           sigma_color=75, sigma_space=75):
    """
    Applies a bilateral filter and returns (original, filtered).

    Parameters:
      input_path   (str): Path to the image
      kernel_size  (int): Kernel diameter (must be odd)
      sigma_color  (int): Color/intensity smoothing (default: 75)
      sigma_space  (int): Spatial smoothing (default: 75)

    Returns:
      Tuple (original_rgb, filtered_rgb)
    """

    image = cv2.imread(input_path)

    
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
    Saves a NumPy image array `img` to the Downloads folder.
    
    - `name`: Base name of the file  
    - `ext`:  File extension, e.g., "png", "jpg", "tif" (default: "png")
    """

    # 1) Pfad zum Download-Ordner 
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    # 2) kompletten Dateinamen zusammenbauen
    filename = f"{name}.{ext.lstrip('.')}"
    output_path = os.path.join(downloads, filename)
    # 3) sicherstellen, dass es den Ordner gibt
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 4) speichern
    if img.ndim == 2:
        plt.imsave(output_path, img)
    else:
        plt.imsave(output_path, img)

    
def save_image_grey(img, name, ext="png"):
    """
    Saves a grayscale image (2D) in uint8 format to the Downloads folder.
    """
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    filename = f"{name}.{ext.lstrip('.')}"
    output_path = os.path.join(downloads, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sicherstellen, dass das Bild im richtigen Format ist
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8) 

    cv2.imwrite(output_path, img)
    print(f"Image saved to: {output_path}")
    



def display_images(original, name="Image"):
    """show image"""

    plt.figure(figsize=(4, 4))
    plt.imshow(original)
    plt.title(name)
    plt.axis('off')
    plt.show()


def display_images_grey(original, name="Image"):
    """Show grayscale image"""

    plt.figure(figsize=(4, 4))
    plt.imshow(original, cmap='gray')
    plt.title(name)
    plt.axis('off')
    plt.show()




def split_channels(img_array):
     
     """Splits an RGB image into its R, G, B channels."""
     
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
    Removes the red channel of an RGB image (sets it to 0)
    and returns an image containing only the green and blue components.

    Parameters:
    rgb_image (np.ndarray): RGB image in the format (H, W, 3)

    Returns:
    np.ndarray: RGB image with the red channel set to 0
"""

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Input image must have shape (H, W, 3) (RGB).")

    # Kopiert das Bild, um das Original nicht zu verändern
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
    Normalizes an RGB image (NumPy array) to the range [0, 1].
    Returns a new float32 array.
    """

    # Muss NumPy-Array mit 3 Dimensionen und 3 Farbkanälen sein
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
    Performs a z-transformation (standardization) on an RGB image,
    leaving channels with σ = 0 unchanged (they remain 0).
    """

    # Eingabeprüfung:
    if not isinstance(rgb_image, np.ndarray):  
        raise TypeError("Eingabebild muss ein NumPy-Array sein.")
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Eingabebild muss die Form (H, W, 3) haben (RGB).")

    # Bild in float32 konvertieren 
    img_float = rgb_image.astype(np.float32, copy=True)

    # Pixelmatrix (H*W, 3)
    flat = img_float.reshape(-1, 3) # reshape image into a list of RGB pixels (one pixel per row)

    # Mittelwert und Standardabweichung berechnen
    means = flat.mean(axis=0) # Mittelwert für jeden Kanal
    stds  = flat.std(axis=0, ddof=0) # Standardabweichung für jeden Kanal
    # "Schutz" gegen Division durch 0
    # Ersetzen von std == 0 durch 1 → verhindert division durch 0
    safe_stds = np.where(stds == 0, 1.0, stds)  

   # Form anpassen, damit jeder Farbkanal auf alle Pixel angewendet werden kann

    means_reshaped = means.reshape((1, 1, 3)) 
    stds_reshaped  = safe_stds.reshape((1, 1, 3))  # jetzt safe_stds statt stds

    # Z-Transformation mit den geschützten Standardabweichungen
    z_image = (img_float - means_reshaped) / stds_reshaped  # Division durch safe_stds

    return z_image






def save_as_numpy(array: np.ndarray, name: str):
    """
    Saves any NumPy array to the Downloads folder.

    Parameters:

    array : np.ndarray  
        The NumPy array to be saved (e.g., a z-transformed image).  
    name : str  
        Base name of the output file (without extension). The function automatically adds ".npy". """

    # 1) Pfad zum aktuellen Downloads-Ordner 
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    
    # 2) Stellt sicher, dass der Downloads-Ordner existiert 
    os.makedirs(downloads_folder, exist_ok=True)
    
    # 3) Dateiname mit .npy-Endung
    filename = name if name.lower().endswith(".npy") else f"{name}.npy"
    
    # 4) Vollständiger Pfad zur Ausgabedatei
    output_path = os.path.join(downloads_folder, filename)
    
    # 5) Array als .npy speichern
    np.save(output_path, array)
    
    print(f"Array gespeichert: {output_path}")
    return output_path





def apply_gaussian_to_array(image_array, kernel_size=5):
    """
    Applies a Gaussian filter to an array.

    Returns: uint8 array (0–255)
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
    Applies a bilateral filter to an array.

    Returns: uint8 array (0–255)
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
    Applies a median filter to the given image.

    Parameters:
    - image: np.ndarray  
    - kernel: int, the size of the median kernel (must be odd)
    """
    #testet ob der Kernel eine ungerade Zahl ist
    if kernel % 2 == 0 or kernel < 1:
        raise ValueError("Kernelgröße muss eine ungerade positive Zahl sein")
    
    filtered = cv2.medianBlur(image, kernel)
    return filtered



def apply_gauß_to_string(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Applies a Gaussian filter with a given kernel size to an image.

    Returns:
    - Filtered image as a NumPy array
    """

    if kernel_size % 2 == 0:

        raise ValueError("Kernel-Größe muss ungerade sein ")

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)




def apply_bilateral_to_string(image: np.ndarray, kernel_size: int, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Applies a bilateral filter with the given kernel size to an image.

    Parameters:
    - image: Input image as a NumPy array  
    - kernel_size: Diameter of the pixel neighborhood (must be odd)  
    - sigma_color: Filter strength for color differences (default: 75)  
    - sigma_space: Filter strength for spatial proximity (default: 75)

    Returns:
    - Filtered image as a NumPy array
    """

    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("Kernel-Größe muss eine positive ungerade Zahl sein")

    # d in bilateralFilter ist der Durchmesser der Nachbarschaft
    filtered_image = cv2.bilateralFilter(image, d=kernel_size, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    return filtered_image


def remove_alpha_channel(image):
    """    Removes the alpha channel from an RGBA image and returns an RGB image."""
 
    if image.ndim == 3 and image.shape[2] == 4: 
        return image[:, :, :3]  # RGBA → RGB
    elif image.ndim == 3 and image.shape[2] == 3:
        return image  # RGB bleibt 
    

# --- KI-generated code START --- DeepSeek AI ---

def apply_watershed(image, num_markers):
    """
    Applies marker-based Watershed segmentation to an image.

    Returns:
        labels_ws (ndarray): Segmented image with labeled regions."""

    # 1. Alpha-Kanal entfernen 
    img_rgb = np.copy(image)
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 4: # RGBA → RGB
        img_rgb = img_rgb[:, :, :3]
    elif image.ndim == 3 and image.shape[2] == 3: # RGB bleibt
        return image 

    # 2. In Graustufen umwandeln
    img_gray = color.rgb2gray(img_rgb) if img_rgb.ndim == 3 else img_rgb # Graustufenbild bleibt Graustufenbild

    # 3. Binärmaske erzeugen 
    th = filters.threshold_otsu(img_gray) # Otsu-Schwellenwert
    binary = img_gray > th

    # 4. Distanz-Transform 
    distance = ndi.distance_transform_edt(binary)   # 4. Distanz-Transformation: misst Abstand jedes Objektpixels zum nächsten Hintergrund
                                                    # → ergibt eine Art "Höhenkarte", die später für die Marker im Watershed genutzt wird


    # 5. Marker setzen 
    coords = peak_local_max(
        distance, 
        labels=binary,
        num_peaks=num_markers,
        footprint=np.ones((15, 15)) # peak_local_max sucht die höchsten Punkte in der Distanzkarte
    )
    markers = np.zeros(distance.shape, dtype=np.int32) # für die Marker wird ein leeres Array erstellt
    # Setze eindeutige Marker an den gegebenen Koordinaten 
    for idx, (r, c) in enumerate(coords, start=1): 
        markers[r, c] = idx # idx = 1, 2, ..., num_markers

    #  6. Watershed anwenden
        # - Nutzt negative Distanzkarte (damit Zentren als Startpunkte dienen)
        # - Marker geben Startpunkte der Regionen an
        # - Nur Bereiche innerhalb der Maske werden segmentiert
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
ax2.imshow(color.label2rgb(
    labels_ws,
    bg_label=0,
    bg_color=(0, 0, 0),
    kind='overlay',
    image_alpha=0  # (optional, aber verstärkt den Effekt)))



# Segmentierte Zellen als Graustufenbild anzeigen
    ax2.imshow(color.label2gray(labels_ws), cmap='gray')
    ax2.axis('on')
    plt.tight_layout()
    plt.show()
    print(f"Anzahl der segmentierten Zellen: {len(np.unique(labels_ws)) - 1}")

    return labels_ws

# --- KI-generated code ENDE --- DeepSeek AI ---



def save_raw_image(img, name, ext="tiff"):
    """
    Save  a raw image (e.g., TIFF) to the Downloads folder.
    """
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    filename = f"{name}.{ext}"
    output_path = os.path.join(downloads, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    iio.imwrite(output_path, img)  # kein Clip, kein Umwandeln
    print(f"RAW image saved to: {output_path}")
