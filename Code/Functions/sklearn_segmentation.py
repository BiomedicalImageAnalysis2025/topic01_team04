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







