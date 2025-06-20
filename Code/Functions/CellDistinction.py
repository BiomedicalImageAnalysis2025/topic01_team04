import numpy as np
import matplotlib.pyplot as plt
import os # To save images
from matplotlib import colors # To convert image models

from Functions.FinalKMeans import init_centroids
from Functions.FinalKMeans import assign_to_centroids
from Functions.FinalKMeans import update_centroids
from Functions.FinalKMeans import save_image
from Functions.FinalKMeans import save_image_universal

def preprocess_gray_with_coords(data, mask=None):
    """
    Erstellt Feature-Vektoren aus Grauwert und (x, y)-Koordinaten.
    Optional: mask = nur für Vordergrund (z.B. Zellen).
    """
    h, w = data.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    # Normalisiere Koordinaten und Intensität
    X = X / w
    Y = Y / h
    I = data / data.max()
    if mask is not None:
        features = np.stack([I[mask], X[mask], Y[mask]], axis=1)
    else:
        features = np.stack([I.ravel(), X.ravel(), Y.ravel()], axis=1)
    return features


#Segmentation with KMeans clustering using just created functions
def kmeans_with_coords(data, k, max_iters=100, tol=1e-4, init_method='kmeans++', space='rgb'):
    """
    Vollständiger K-Means Ablauf:
    1. init_centroids
    2. assign_to_centroids
    3. update_centroids
    4. Abbruch bei Konvergenz oder max_iters
    Returns: centroids, labels, segmented_image
    - data: 2D-Array (n_samples, n_features) für RGB/HSV/Grayscale
    - k: Anzahl der Cluster 
    """

    #Normalize data
    data = np.copy(data.astype(float))
    data = (data - data.min()) / (data.max() - data.min())

    #Drop alpha channel if present
    if data.ndim == 3 and data.shape[2] == 4:
        data = data[..., :3]
    else:
        data = data

     # threshold um den hintergrund schwarz zu färben?
    threshold = 0.1
    mask = data > threshold
    
    img = preprocess_gray_with_coords(data, mask=mask)

    data_shape = data.shape
    centroids = init_centroids(img, k, method=init_method)
    for i in range(max_iters):
        labels = assign_to_centroids(img, centroids)
        new_centroids = update_centroids(img, labels, k)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

        segmented_image = reconstruct_colored_segmentation(labels, mask, data_shape, k)
    return centroids, labels, segmented_image


def reconstruct_colored_segmentation(labels, mask, shape, k):
    """
    Rekonstruiert ein farbiges Segmentierungsbild mit schwarzem Hintergrund.
    """
    color_map = plt.cm.get_cmap('tab10', k)
    colors = color_map(np.arange(k))[:, :3]  # RGB-Farben für Cluster
    seg_img = np.zeros((shape[0], shape[1], 3))  # Schwarz als Hintergrund
    seg_img[mask] = colors[labels]
    return seg_img