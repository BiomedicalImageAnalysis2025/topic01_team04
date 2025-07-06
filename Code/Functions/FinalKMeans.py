# Coded by Jonas Schenker
# Code for FinalKMeans.ipynb

import numpy as np
import matplotlib.pyplot as plt
import os # To save images
from matplotlib import colors # To convert image models

# K-Means clustering for image segmentation in RGB, HSV, and Grayscale
# Modular implementation with init_centroids, assign_to_centroids, update_centroids


# Helper functions for image preprocessing
def preprocess_rgb(img):
    #h, w, _ = img.shape
    return img.reshape(-1, 3)#, (h, w)

def preprocess_grayscale(img):
    if img.ndim == 3:
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]) # converts RGB image into grayscale
    else:
        gray = img
    return gray.reshape(-1, 1)#, gray.shape


def preprocess_hsv(img):
    """
    Converts an RGB image to HSV if necessary and returns the HSV features.
    If the image is already in HSV (values between 0 and 1), nothing is changed.
    """
    # Check if the image is RGB (typically uint8 or float with max > 1)
    if img.shape[-1] == 3 and (img.dtype == np.uint8 or img.max() > 1.0):
        # Image is RGB, convert to float and normalize
        img = img.astype(float)
        if img.max() > 1.0:
            img /= 255.0
        hsv = colors.rgb_to_hsv(img)
    else:
        # image is already in HSV format
        hsv = img
    #h, w, _ = hsv.shape
    return hsv.reshape(-1, 3)#, (h, w)


def preprocess_image(img, space='rgb'):
    """
    Preprocessing the image data depending on the color space.
    - space='rgb': RGB image
    - space='hsv': HSV image
    - space='gray': Grayscale image
    """
    if space == 'rgb':
        return preprocess_rgb(img)
    elif space == 'hsv':
        return preprocess_hsv(img)
    elif space == 'gray':
        return preprocess_grayscale(img)
    


# Functions for segmentation
def init_centroids(data, k, method='random'):
    """
    Initializes k centroids from the data.
    - method='random': Random selection
    - method='kmeans++': k-means++ seeding (Arthur & Vassilvitskii, 2007).
    """
    n_samples = data.shape[0]
    if method == 'kmeans++':
        centroids = []
        # first centroid randomly
        idx = np.random.randint(n_samples)
        centroids.append(data[idx])
        # additional centroids based on distance
        for _ in range(1, k):
            dists = np.min(np.linalg.norm(data[:, None, :] - np.array(centroids)[None, :, :], axis=2)**2, axis=1)
            probs = dists / dists.sum()
            idx = np.random.choice(n_samples, p=probs)
            centroids.append(data[idx])
        return np.vstack(centroids).astype(float)
    else:
        idx = np.random.choice(n_samples, k, replace=False)
        return data[idx].astype(float)
    

def assign_to_centroids(data, centroids):
    """
    Assigns each sample to the nearest centroid.
    Distance metric: Euclidean
    Returns: labels (n_samples,)
    - centroids: 2D array with shape (k, n_features)
    """
    dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def update_centroids(data, labels, k):
    """
    Computes new centroids as the mean of the data points assigned to each cluster.
    - labels: 1D array with cluster assignments (e.g., shape (n_samples,))
    """
    n_features = data.shape[1]
    centroids = np.zeros((k, n_features), dtype=float)
    for j in range(k):
        pts = data[labels == j]
        if len(pts) > 0:
            centroids[j] = pts.mean(axis=0)
        else:
            centroids[j] = data[np.random.randint(data.shape[0])]
    return centroids



#Segmentation with KMeans clustering using just created functions
def kmeans(data, k, max_iters=100, tol=1e-4, init_method='kmeans++', space='rgb'):
    """
    Complete K-Means workflow:
    1. init_centroids
    2. assign_to_centroids
    3. update_centroids
    4. Stop on convergence or max_iters
    Returns: centroids, labels, segmented_image
    - data: 2D array (n_samples, n_features) for RGB/HSV/Grayscale
    - k: number of clusters 
    - max_iters: maximum number of iterations
    - tol: tolerance for convergence, if the change in centroids is less than tol, stop
    - init_method: 'random' or 'kmeans++' for centroid initialization
    - space: 'rgb', 'hsv', or 'gray' for color space of the image
    """
    #Normalize data
    data = np.copy(data.astype(float))
    data = (data - data.min()) / (data.max() - data.min())

    #Drop alpha channel if present
    if data.ndim == 3 and data.shape[2] == 4:
        data = data[..., :3]
    else:
        data = data
    
    img = preprocess_image(data, space=space)
    data_shape = data.shape
    centroids = init_centroids(img, k, method=init_method)
    for i in range(max_iters):
        labels = assign_to_centroids(img, centroids)
        new_centroids = update_centroids(img, labels, k)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

        segmented_image = reconstruct_segmented_image(centroids, labels, data_shape, space)
    return centroids, labels, segmented_image



#Reconstruct just segmented images
def reconstruct_segmented_image(centroids, labels, data_shape, space):
    """
    Reconstructs a segmented image from KMeans labels and centroids.

    Parameters:
    - labels: 1D array with cluster assignment for each pixel (e.g., shape (h*w,))
    - centroids: Array with centroids (e.g., shape (k, 1) for grayscale or (k, 3) for RGB)
    - data_shape: Tuple with the target image shape (e.g., (h, w) for grayscale, (h, w, 3) for RGB)
    - space: 'rgb', 'hsv', or 'gray' to determine the color space of the image

    Returns:
        segmented_image: The reconstructed segmented image in original shape.
    """
    
    segmented_flat = centroids[labels]
    if space == 'rgb':
        # assign each pixel the RGB color of its cluster
        segmented_image = segmented_flat.reshape(data_shape[0], data_shape[1], 3)
    elif space == 'hsv':
        # assign each pixel the HSV color of its cluster and convert to RGB
        segmented_image = colors.hsv_to_rgb(segmented_flat.reshape(data_shape[0], data_shape[1], 3))
    elif space == 'gray':
        # assign each pixel the grayscale value of its cluster
        segmented_image = segmented_flat.reshape(data_shape[0], data_shape[1], 1)
      
    return segmented_image


#Save image
def save_image(image, path):
    """
    saves an image (numpy array) to the specified path.
    """
    # create folder if it does not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, image)

#Save image of different color models (RGB, HSV, Grayscale)
def save_image_universal(image, path, space='rgb'):
    """
    Saves an image according to the color space (RGB, HSV, Grayscale) at the specified path.
    - image: numpy array
    - path: save path (including filename)
    - space: 'rgb', 'hsv', or 'gray'
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.copy(image)
    if space == 'hsv':
        # convert HSV to RGB if necessary
        #if img.max() > 1.0:
         #   img = img / 255.0
        #img = colors.hsv_to_rgb(img)  #conversion already done in reconstruct_segmented_image done
        plt.imsave(path, img)
    elif space == 'rgb':
        if img.max() > 1.0:
            img = img / 255.0
        plt.imsave(path, img)
    elif space == 'gray':
        if img.max() > 1.0:
            img = img / 255.0
        # if image is 3D with single channel, squeeze it to 2D
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(axis=2)
        plt.imsave(path, img, cmap='gray')


# Function to identify the ideal number of clusters using the Elbow Method
def elbow_method(data, max_k=10, max_iters=100, tol=1e-4, init_method='kmeans++', space='rgb'):
    """
    Identifies the ideal number of clusters using the Elbow Method.

    Parameters:
    - image: 3D numpy array representing the image.
    - max_k: Maximum number of clusters to test.
    - max_iters: Maximum number of iterations for K-Means.
    - tol: Tolerance for convergence, if the change in centroids is less than tol, stop.
    - init_method: 'random' or 'kmeans++' for centroid initialization.
    - space: 'rgb', 'hsv', or 'gray' for color space of the image.

    Returns:
    - wcss: List of WCSS values for each k.
    """
    wcss = []

    #Drop alpha channel if present
    if data.ndim == 3 and data.shape[2] == 4:
        data = data[..., :3]
    else:
        data = data
        
    reshaped_image = data.reshape(-1, 3)
    for k in range(1, max_k + 1):
        centroids, labels, _ = kmeans(data, k, max_iters, tol, init_method, space)
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


# Function to identify the k where the elbow occurs (The "elbow" is the point farthest from the line connecting the first and last point ("knee point" using the "distance to line" method).)
def find_elbow(wcss):
    """
    Identifies the elbow point in the WCSS values using the 'distance to line' method. (DSPA2: Data Science and Predictive Analytics (UMich HS650), VIII. Unsupervised Clustering.)

    Parameters:
    - wcss: List of WCSS values for each k.

    Returns:
    - elbow: The k value where the elbow occurs.
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

 