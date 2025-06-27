#Sklearn KMeans clustering for segmentation of images
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans


from sklearn.cluster import KMeans
import numpy as np

def skcluster_image(image_array, n_clusters=2):
    """
    Führt KMeans-Clustering auf einem Bild durch.
    Unterstützt sowohl 2D (z.B. Hue) als auch 3D (z.B. RGB) Bilder.

    Parameter:
    - image_array: NumPy-Array des Bildes (z.B. von plt.imread geladen), 2D oder 3D
    - n_clusters: Anzahl der Cluster (Farben)

    Rückgabe:
    - clustered_img: Clustertes Bild als NumPy-Array (gleiches Format wie input)
    """
    original_shape = image_array.shape
    
    if len(original_shape) == 2:
        # 2D Bild (z.B. Hue), reshape zu (Pixelanzahl, 1)
        pixels = image_array.reshape(-1, 1)
    elif len(original_shape) == 3:
        # 3D Bild (z.B. RGB), reshape zu (Pixelanzahl, Kanäle)
        pixels = image_array.reshape(-1, original_shape[2])
    else:
        raise ValueError("Unsupported image array shape: must be 2D or 3D")

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
    kmeans.fit(pixels)

    clustered = kmeans.cluster_centers_[kmeans.labels_]
    clustered_img = clustered.reshape(original_shape)
    
    return clustered_img





def save_image(img, name, ext="tiff"):

    # 1) Pfad zum Download-Ordner (Windows, macOS, Linux)
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    # 2) kompletten Dateinamen zusammenbauen
    filename = f"{name}.{ext.lstrip('.')}"
    output_path = os.path.join(downloads, filename)
    # 3) sicherstellen, dass es den Ordner gibt
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 4) speichern
    success = cv2.imwrite(output_path, img)
    if success:
        print(f"Bild gespeichert unter: {output_path}")
    else:
        print(f"❌ Fehler beim Speichern von: {output_path}")



import numpy as np
from sklearn.cluster import KMeans

def skcluster_watershed(image_array, n_clusters=2):
    """
    Führt KMeans-Clustering auf einem Graustufen- (2D) oder Farb- (3D) Bild durch.
    Für Watershed-Bilder in Graustufen (2D) funktioniert das direkt.
    
    Parameter:
    - image_array: NumPy-Array des Bildes (2D Graustufen oder 3D Farb)
    - n_clusters: Anzahl der Cluster (Farben)
    
    Rückgabe:
    - clustered_img: Bild mit Pixeln auf die jeweiligen Cluster-Zentren gerundet,
      gleiche Form und Typ wie das Input-Bild (float->float, uint8->uint8)
    """
    original_shape = image_array.shape
    
    # 2D Graustufen: reshape zu (Pixelanzahl, 1)
    if len(original_shape) == 2:
        pixels = image_array.reshape(-1, 1)
    # 3D Farb: reshape zu (Pixelanzahl, Kanäle)
    elif len(original_shape) == 3:
        pixels = image_array.reshape(-1, original_shape[2])
    else:
        raise ValueError("Unsupported image shape: only 2D or 3D arrays supported.")
    
    # Falls Bild float mit 0..1, für KMeans besser auf 0..255 skalieren (optional)
    if pixels.dtype == np.float32 or pixels.dtype == np.float64:
        if pixels.max() <= 1.0:
            pixels = pixels * 255
    
    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans.fit(pixels)
    
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    clustered_img = clustered.reshape(original_shape)
    
    # Rückskalierung falls nötig (float)
    if image_array.dtype == np.uint8:
        clustered_img = np.clip(clustered_img, 0, 255).astype(np.uint8)
    else:
        clustered_img = clustered_img.astype(image_array.dtype)
    
    return clustered_img


import numpy as np
from sklearn.cluster import KMeans

def skcluster_hsv_image(hsv_image, n_clusters=2):
    """
    Führt KMeans-Clustering auf einem HSV-Bild durch.
    Dabei wird der Hue-Kanal zirkulär in Sin und Cos umgewandelt,
    um den Farbton korrekt abzubilden.

    Parameter:
    - hsv_image: NumPy-Array (HxWx3), mit Kanälen [Hue, Saturation, Value].
                 Hue wird erwartet als normierter Wert [0,1], entspricht 0°-360°.
    - n_clusters: Anzahl der Cluster.

    Rückgabe:
    - clustered_img: NumPy-Array, gleiche Form wie hsv_image,
                     mit den Cluster-Zentren (Features) als Pixelwerte.
    """

    # Sicherstellen, dass hsv_image float ist
    hsv = hsv_image.astype(np.float32)
    
    # Hue (0-1) in Winkel (0-2pi)
    hue_angle = hsv[:, :, 0] * 2 * np.pi
    
    # Hue zirkulär als Sin und Cos
    hue_sin = np.sin(hue_angle)
    hue_cos = np.cos(hue_angle)
    
    # Saturation und Value bleiben wie sie sind
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    
    # Feature-Matrix vorbereiten (HxWx4)
    features = np.stack([hue_sin, hue_cos, sat, val], axis=-1)
    
    # Umformen in (Pixelanzahl x Featureanzahl)
    pixels = features.reshape(-1, 4)
    
    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
    kmeans.fit(pixels)
    
    # Clusterzentren zurück als Bild (Pixelweise Clusterzentren-Werte)
    clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]
    clustered_img = clustered_pixels.reshape(hsv.shape[0], hsv.shape[1], 4)
    
    # Optional: zurück zu "normalem" HSV (Hue als Winkel aus Sin & Cos)
    # Hue neu berechnen:
    new_hue = (np.arctan2(clustered_img[:, :, 0], clustered_img[:, :, 1]) / (2 * np.pi)) % 1.0
    new_sat = clustered_img[:, :, 2]
    new_val = clustered_img[:, :, 3]
    
    # Endbild HSV (HxWx3)
    clustered_hsv = np.stack([new_hue, new_sat, new_val], axis=-1)
    
    return clustered_hsv


import os
import matplotlib.pyplot as plt
from skimage.io import imread
from cellpose import models, plot
from skimage.io import imwrite

# Deine save_image_gt-Funktion
def save_image_gt(img, name, ext="tiff"):
    # Pfad zum Downloads-Ordner (macOS)
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    filename = f"{name}.{ext.lstrip('.')}"
    output_path = os.path.join(downloads, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        # Speichern in 16-bit, um Zell-IDs nicht zu verlieren
        imwrite(output_path, img.astype("uint16"))
        print(f"✅ Bild gespeichert unter: {output_path}")
    except Exception as e:
        print(f"❌ Fehler beim Speichern von: {output_path}")
        print(e)




