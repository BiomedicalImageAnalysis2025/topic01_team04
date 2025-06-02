import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

def sklearn(image_array, n_clusters=2):
    """
    Führt KMeans-Clustering auf einem Bild durch.

    Parameter:
    - image_array: NumPy-Array des Bildes (z. B. von plt.imread geladen)
    - n_clusters: Anzahl der Cluster (Farben)

    Rückgabe:
    - clustered_img: Clustertes Bild als NumPy-Array (gleiches Format wie input)
    """
    original_shape = image_array.shape
    pixels = image_array.reshape(-1, image_array.shape[2])
    
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
    kmeans.fit(pixels)
    
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    clustered_img = clustered.reshape(original_shape)
    
    return clustered_img


