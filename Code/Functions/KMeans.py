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

centroids = init_centroids(3)
centroids

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