import numpy as np
import matplotlib.pyplot as plt


#Function to extract the single colour channels of the image (wahrscheinlich unnötig)
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


# Function to pick random centroids for K-Means clustering (range 0-1 --> normalized data)
def init_centroids(k):
    
   centroids = np.random.rand(k, 3)  # RGB-Werte zwischen 0 und 1
   return centroids


#(
# Function to calculate euclidean distance in a 3D space
def euclidean_distance_3d(p1, p2):
    """
    Berechnet die euklidische Distanz zwischen zwei Punkten im 3D-Raum.
    
    Parameter:
    - p1 (array-like): Erster Punkt (z. B. [R, G, B])
    - p2 (array-like): Zweiter Punkt (z. B. [R, G, B])
    
    Rückgabe:
    - float: Die euklidische Distanz
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(np.sum((p1 - p2) ** 2))
#) --> already implemented in assign_to_centroids


# Function to assign each pixel to the nearest centroid
def assign_to_centroids(pixels, centroids):
    """
    Weist jedem Pixel den Index des nächstgelegenen Zentroiden zu.
    
    Parameter:
    - pixels (ndarray): (n, 3) RGB-normalisierte Pixel
    - centroids (ndarray): (k, 3) aktuelle Zentroiden
    
    Rückgabe:
    - labels (ndarray): (n,) Array mit Index des jeweils nächsten Zentroids
    """
    n = pixels.shape[0]
    k = centroids.shape[0]
    labels = np.empty(n, dtype=int)
    
    for i in range(n):
        # Berechne euklidische Distanz zu allen k Zentroiden
        distances = np.sqrt(np.sum((centroids - pixels[i])**2, axis=1))
        labels[i] = np.argmin(distances)
    
    return labels


# Function to update centroids based on assigned pixels
def update_centroids(pixels, labels, k):
    """
    Aktualisiert die Zentroiden basierend auf den zugewiesenen Pixeln.
    
    Parameter:
    - pixels (ndarray): (n, 3) RGB-normalisierte Pixel
    - labels (ndarray): (n,) Array mit Index des jeweils nächsten Zentroids
    - k (int): Anzahl der Zentroiden
    
    Rückgabe:
    - new_centroids (ndarray): (k, 3) aktualisierte Zentroiden
    """
    new_centroids = np.zeros((k, 3))
    
    for i in range(k):
        # Extrahiere alle Pixel, die dem i-ten Zentroiden zugewiesen sind
        assigned_pixels = pixels[labels == i]
        
        if len(assigned_pixels) > 0:
            new_centroids[i] = np.mean(assigned_pixels, axis=0)
        else:
            new_centroids[i] = np.random.rand(3)  # Zufälliger Wert, falls kein Pixel zugewiesen ist
    
    return new_centroids


# Function to perform K-Means clustering
def kmeans_clustering(image, k, max_iterations=100):
    """
    Perform K-means clustering on a 3D image.

    Parameters:
    - image: 3D numpy array representing the image.
    - k: Number of clusters.
    - max_iterations: Maximum number of iterations for convergence.

    Returns:
    - segmented_image: 3D numpy array with clustered pixel values.
    """
    # Reshape Image
    reshaped_image = image.reshape(-1, 3)

    # Initialize centroids
    centroids = init_centroids(k)

    for _ in range(max_iterations):
        # Assign pixels to the nearest centroid
        labels = assign_to_centroids(reshaped_image, centroids)

        # Update centroids based on the assigned pixels
        new_centroids = update_centroids(reshaped_image, labels, k)

        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    # Create segmented image based on final labels
    labels_2d = labels.reshape(image.shape[0], image.shape[1])
    segmented_image = np.zeros_like(image)
    for i in range(k):
        segmented_image[labels_2d == i] = centroids[i]

    return segmented_image