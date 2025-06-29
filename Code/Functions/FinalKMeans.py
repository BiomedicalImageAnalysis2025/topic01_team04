import numpy as np
import matplotlib.pyplot as plt
import os # To save images
from matplotlib import colors # To convert image models


# K-Means Clustering für Bildsegmentierung in RGB, HSV und Grayscale
# Modular implementiert mit init_centroids, assign_to_centroids, update_centroids


# Hilfsfunktionen für Bildvorbereitung
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
    Wandelt ein RGB-Bild in HSV um, falls nötig, und gibt die HSV-Features zurück.
    Falls das Bild bereits HSV ist (Wertebereich H,S,V jeweils zwischen 0 und 1), wird nichts geändert.
    """
    # Prüfe, ob das Bild RGB ist (typischerweise uint8 oder float mit max > 1)
    if img.shape[-1] == 3 and (img.dtype == np.uint8 or img.max() > 1.0):
        # Bild ist RGB, umwandeln in float und normalisieren
        img = img.astype(float)
        if img.max() > 1.0:
            img /= 255.0
        hsv = colors.rgb_to_hsv(img)
    else:
        # Bild ist vermutlich schon HSV
        hsv = img
    #h, w, _ = hsv.shape
    return hsv.reshape(-1, 3)#, (h, w)


def preprocess_image(img, space='rgb'):
    """
    Preprocessing der Bilddaten je nach Farbraum.
    - space='rgb': RGB-Bild
    - space='hsv': HSV-Bild
    - space='gray': Graustufenbild
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
    Initialisiert k Zentroiden aus den Daten.
    - method='random': Zufällige Auswahl (MacQueen, 1967).
    - method='kmeans++': k-means++ Seeding (Arthur & Vassilvitskii, 2007).
    """
    n_samples = data.shape[0]
    if method == 'kmeans++':
        centroids = []
        # erstes Zentrum zufällig
        idx = np.random.randint(n_samples)
        centroids.append(data[idx])
        # weitere Zentren
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
    Ordnet jedes Sample dem nächsten Zentroiden zu.
    Distanzmetriken: euklidisch (Lloyd, 1982).
    Returns: labels (n_samples,)
    """
    dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def update_centroids(data, labels, k):
    """
    Berechnet neue Zentroiden als Mittelwerte der jeweils zugeordneten Datenpunkte.
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
    Rekonstruiert ein segmentiertes Bild aus KMeans-Labels und Zentroiden.
    
    Parameters:
        labels: 1D-Array mit Clusterzuordnung für jedes Pixel (z.B. shape (h*w,))
        centroids: Array mit den Zentroiden (z.B. shape (k, 1) für Graustufen oder (k, 3) für RGB)
        image_shape: Tuple mit der Zielbildform (z.B. (h, w) für Graustufen, (h, w, 3) für RGB)
    
    Returns:
        segmented_image: Das rekonstruierte segmentierte Bild in Originalform.
    """
    
    segmented_flat = centroids[labels]
    if space == 'rgb':
        # Weise jedem Pixel die Farbe seines Clusters zu
        segmented_image = segmented_flat.reshape(data_shape[0], data_shape[1], 3)
    elif space == 'hsv':
        # Weise jedem Pixel die Farbe seines Clusters zu und konvertiere zurück nach RGB
        segmented_image = colors.hsv_to_rgb(segmented_flat.reshape(data_shape[0], data_shape[1], 3))
    elif space == 'gray':
        # Weise jedem Pixel die Graustufenfarbe seines Clusters zu
        segmented_image = segmented_flat.reshape(data_shape[0], data_shape[1], 1)
      
    return segmented_image


#Save image
def save_image(image, path):
    """
    Speichert ein Bild (numpy array) im angegebenen Pfad.
    """
    # Erstelle den Ordner, falls er nicht existiert
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, image)

#Save image of different color models (RGB, HSV, Grayscale)
def save_image_universal(image, path, space='rgb'):
    """
    Speichert ein Bild je nach Farbraum (RGB, HSV, Grayscale) am angegebenen Pfad.
    - image: numpy array
    - path: Speicherpfad (inkl. Dateiname)
    - space: 'rgb', 'hsv' oder 'gray'
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.copy(image)
    if space == 'hsv':
        # HSV zu RGB konvertieren
        #if img.max() > 1.0:
         #   img = img / 255.0
        #img = colors.hsv_to_rgb(img)  #Umwandlung bereits in reconstruct_segmented_image durchgeführt
        plt.imsave(path, img)
    elif space == 'rgb':
        if img.max() > 1.0:
            img = img / 255.0
        plt.imsave(path, img)
    elif space == 'gray':
        if img.max() > 1.0:
            img = img / 255.0
        # Falls das Bild shape (H, W, 1) hat, squeeze auf (H, W)
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

 