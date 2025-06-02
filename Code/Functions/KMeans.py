import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


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


# Function to perform K-Means clustering in RGB space
def kmeans_clusteringRGB(image, k, max_iterations=100, return_labels_centroids=False):
    """
    Perform K-means clustering on a 3D image.

    Parameters:
    - image: 3D numpy array representing the image.
    - k: Number of clusters.
    - max_iterations: Maximum number of iterations for convergence.

    Returns:
    - segmented_image: 3D numpy array with clustered pixel values.
    - (optional) labels: 1D array of cluster assignments.
    - (optional) centroids: 2D array of centroid values.
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

    if return_labels_centroids:
        return segmented_image, labels, centroids

    return segmented_image


 #Function to perform K-Means clustering in HSV space
def kmeans_clusteringHSV(image, k, max_iterations=100, return_labels_centroids=False):
    """
    Perform K-means clustering on a 3D image in HSV space.

    Parameters:
    - image: 3D numpy array representing the image.
    - k: Number of clusters.
    - max_iterations: Maximum number of iterations for convergence.

    Returns:
    - segmented_image: 3D numpy array with clustered pixel values in HSV space.
    - (optional) labels: 1D array of cluster assignments.
    - (optional) centroids: 2D array of centroid values in HSV space.
    """
    # Convert RGB to HSV
    hsv_image = plt.colors.rgb_to_hsv(image / 255.0)  # Normalize to [0, 1] for conversion

    # Reshape Image
    reshaped_image = hsv_image.reshape(-1, 3)

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
    segmented_image = np.zeros_like(hsv_image)
    for i in range(k):
        segmented_image[labels_2d == i] = centroids[i]

    if return_labels_centroids:
        return segmented_image, labels, centroids

    return segmented_image


 #Function to perform K-Means clustering in Intensity space for grayscale images
def kmeans_clusteringGrayscale(image, k, max_iterations=100, return_labels_centroids=False):
    """
    Perform K-means clustering on a grayscale image.

    Parameters:
    - image: 2D numpy array representing the grayscale image.
    - k: Number of clusters.
    - max_iterations: Maximum number of iterations for convergence.

    Returns:
    - segmented_image: 2D numpy array with clustered pixel values.
    - (optional) labels: 1D array of cluster assignments.
    - (optional) centroids: 1D array of centroid values.
    """
    # Reshape Image
    reshaped_image = image.reshape(-1, 1)

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

    if return_labels_centroids:
        return segmented_image, labels, centroids

    return segmented_image


# Function to identify the ideal number of clusters using the Elbow Method
def elbow_method(image, max_k=10):
    """
    Identifies the ideal number of clusters using the Elbow Method.

    Parameters:
    - image: 3D numpy array representing the image.
    - max_k: Maximum number of clusters to test.

    Returns:
    - wcss: List of WCSS values for each k.
    """
    wcss = []
    reshaped_image = image.reshape(-1, 3)
    for k in range(1, max_k + 1):
        _, labels, centroids = kmeans_clusteringRGB(image, k, return_labels_centroids=True)
        # WCSS: Sum of squared distances of each point to its assigned centroid
        distances = np.sum((reshaped_image - centroids[labels]) ** 2)
        wcss.append(distances)
    return wcss


# Function to plot the Elbow Method results
def plot_elbow_method(wcss):
    """
    Plots the WCSS values to visualize the Elbow Method.

    Parameters:
    - wcss: List of WCSS values for each k.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.xticks(range(1, len(wcss) + 1))
    plt.grid()
    plt.show()


# Function to identify the k where the elbow occurs (Das "Knie" ist der Punkt, der am weitesten von der Verbindungslinie zwischen erstem und letztem Punkt entfernt ist ("knee point" nach der "distance to line"-Methode).)
def find_elbow(wcss):
    """
    Identifies the elbow point in the WCSS values using the 'distance to line' method.

    Parameters:
    - wcss: List of WCSS values for each k.

    Returns:
    - elbow_k: The k value where the elbow occurs.
    """
    n_points = len(wcss)
    all_k = np.arange(1, n_points + 1)
    # Line from first to last point
    line_vec = np.array([all_k[-1] - all_k[0], wcss[-1] - wcss[0]])
    line_vec = line_vec / np.linalg.norm(line_vec)
    # Distances
    distances = []
    for i in range(n_points):
        point = np.array([all_k[i] - all_k[0], wcss[i] - wcss[0]])
        proj = np.dot(point, line_vec) * line_vec
        dist = np.linalg.norm(point - proj)
        distances.append(dist)
    elbow_index = np.argmax(distances)
    return all_k[elbow_index]







#kDTree approach

def assign_to_centroids_kdtree(pixels, centroids):
    """
    Weist jedem Pixel den Index des nächstgelegenen Zentroiden zu, mit KD-Tree.
    """
    from scipy.spatial import KDTree as KDTree
    tree = KDTree(centroids)
    distances, labels = tree.query(pixels)
    return labels

def kmeans_clusteringRGB_kdtree(image, k, max_iterations=100, return_labels_centroids=False):
    reshaped_image = image.reshape(-1, 3)
    centroids = init_centroids(k)

    for _ in range(max_iterations):
        labels = assign_to_centroids_kdtree(reshaped_image, centroids)
        new_centroids = update_centroids(reshaped_image, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    labels_2d = labels.reshape(image.shape[0], image.shape[1])
    segmented_image = np.zeros_like(image)
    for i in range(k):
        segmented_image[labels_2d == i] = centroids[i]

    if return_labels_centroids:
        return segmented_image, labels, centroids
    return segmented_image

def kmeans_clusteringHSV_kdtree(image, k, max_iterations=100, return_labels_centroids=False):
    """
    Perform K-means clustering on a 3D image in HSV space using KD-Tree.
    """
    hsv_image = plt.colors.rgb_to_hsv(image / 255.0)  # Normalize to [0, 1] for conversion
    reshaped_image = hsv_image.reshape(-1, 3)
    centroids = init_centroids(k)

    for _ in range(max_iterations):
        labels = assign_to_centroids_kdtree(reshaped_image, centroids)
        new_centroids = update_centroids(reshaped_image, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    labels_2d = labels.reshape(image.shape[0], image.shape[1])
    segmented_image = np.zeros_like(hsv_image)
    for i in range(k):
        segmented_image[labels_2d == i] = centroids[i]

    if return_labels_centroids:
        return segmented_image, labels, centroids
    return segmented_image

def kmeans_clusteringGrayscale_kdtree(image, k, max_iterations=100, return_labels_centroids=False):
    """
    Perform K-means clustering on a grayscale image using KD-Tree.

    Parameters:
    - image: 2D numpy array representing the grayscale image.
    - k: Number of clusters.
    - max_iterations: Maximum number of iterations for convergence.

    Returns:
    - segmented_image: 2D numpy array with clustered pixel values.
    - (optional) labels: 1D array of cluster assignments.
    - (optional) centroids: 1D array of centroid values.
    """
    reshaped_image = image.reshape(-1, 1)
    centroids = init_centroids(k)

    for _ in range(max_iterations):
        labels = assign_to_centroids_kdtree(reshaped_image, centroids)
        new_centroids = update_centroids(reshaped_image, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    labels_2d = labels.reshape(image.shape[0], image.shape[1])
    segmented_image = np.zeros_like(image)
    for i in range(k):
        segmented_image[labels_2d == i] = centroids[i]

    if return_labels_centroids:
        return segmented_image, labels, centroids
    return segmented_image

def elbow_method_kdtree(image, max_k=10):
    wcss = []
    reshaped_image = image.reshape(-1, 3)
    for k in range(1, max_k + 1):
        _, labels, centroids = kmeans_clusteringRGB_kdtree(image, k, return_labels_centroids=True)
        distances = np.sum((reshaped_image - centroids[labels]) ** 2)
        wcss.append(distances)
    return wcss