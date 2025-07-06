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
    Performs KMeans clustering on an image.
    Supports both 2D (e.g., Hue) and 3D (e.g., RGB) images.

    Parameters:
    - image_array: NumPy array of the image (e.g., loaded with plt.imread), 2D or 3D
    - n_clusters: Number of clusters (colors)

    Returns:
    - clustered_img: Clustered image as a NumPy array (same format as input)
    """
    original_shape = image_array.shape
    
    if len(original_shape) == 2:
        # 2D image (e.g., Hue), reshape to (number of pixels, 1)
        pixels = image_array.reshape(-1, 1)
    elif len(original_shape) == 3:
        # 3D image (e.g., RGB), reshape to (number of pixels, channels)
        pixels = image_array.reshape(-1, original_shape[2])
    else:
        raise ValueError("Unsupported image array shape: must be 2D or 3D")

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
    kmeans.fit(pixels)

    clustered = kmeans.cluster_centers_[kmeans.labels_]
    clustered_img = clustered.reshape(original_shape)
    
    return clustered_img





def save_image(img, name, ext="tiff"):

    # 1) Path to the Downloads folder (Windows, macOS, Linux)
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    # 2) Build the complete filename
    filename = f"{name}.{ext.lstrip('.')}"
    output_path = os.path.join(downloads, filename)
    # 3) Ensure that the folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 4) Save
    success = cv2.imwrite(output_path, img)
    if success:
        print(f"Image saved at: {output_path}")
    else:
        print(f"❌ Error saving: {output_path}")



import numpy as np
from sklearn.cluster import KMeans

def skcluster_watershed(image_array, n_clusters=2):
    """
    Performs KMeans clustering on a grayscale (2D) or color (3D) image.
    Works directly for watershed images in grayscale (2D).
    
    Parameters:
    - image_array: NumPy array of the image (2D grayscale or 3D color)
    - n_clusters: Number of clusters (colors)
    
    Returns:
    - clustered_img: Image with pixels rounded to their respective cluster centers,
      same shape and type as the input image (float->float, uint8->uint8)
    """
    original_shape = image_array.shape
    
    # 2D grayscale: reshape to (number of pixels, 1)
    if len(original_shape) == 2:
        pixels = image_array.reshape(-1, 1)
    # 3D color: reshape to (number of pixels, channels)
    elif len(original_shape) == 3:
        pixels = image_array.reshape(-1, original_shape[2])
    else:
        raise ValueError("Unsupported image shape: only 2D or 3D arrays supported.")
    
    # If image is float with 0..1, better to scale to 0..255 for KMeans (optional)
    if pixels.dtype == np.float32 or pixels.dtype == np.float64:
        if pixels.max() <= 1.0:
            pixels = pixels * 255
    
    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans.fit(pixels)
    
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    clustered_img = clustered.reshape(original_shape)
    
    # Rescale if necessary (float)
    if image_array.dtype == np.uint8:
        clustered_img = np.clip(clustered_img, 0, 255).astype(np.uint8)
    else:
        clustered_img = clustered_img.astype(image_array.dtype)
    
    return clustered_img


import numpy as np
from sklearn.cluster import KMeans

def skcluster_hsv_image(hsv_image, n_clusters=2):
    """
    Performs KMeans clustering on an HSV image.
    The Hue channel is circularly converted into sine and cosine
    to correctly represent the color hue.

    Parameters:
    - hsv_image: NumPy array (HxWx3), with channels [Hue, Saturation, Value].
                 Hue is expected as normalized value [0,1], corresponding to 0°-360°.
    - n_clusters: Number of clusters.

    Returns:
    - clustered_img: NumPy array, same shape as hsv_image,
                     with cluster centers (features) as pixel values.
    """

    # Ensure hsv_image is float
    hsv = hsv_image.astype(np.float32)
    
    # Hue (0-1) to angle (0-2pi)
    hue_angle = hsv[:, :, 0] * 2 * np.pi
    
    # Hue circular as sine and cosine
    hue_sin = np.sin(hue_angle)
    hue_cos = np.cos(hue_angle)
    
    # Saturation and Value remain as they are
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    
    # Prepare feature matrix (HxWx4)
    features = np.stack([hue_sin, hue_cos, sat, val], axis=-1)
    
    # Reshape to (number of pixels x number of features)
    pixels = features.reshape(-1, 4)
    
    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
    kmeans.fit(pixels)
    
    # Cluster centers back as image (pixelwise cluster center values)
    clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]
    clustered_img = clustered_pixels.reshape(hsv.shape[0], hsv.shape[1], 4)
    
    # Optional: back to "normal" HSV (Hue as angle from Sin & Cos)
    # Recalculate Hue:
    new_hue = (np.arctan2(clustered_img[:, :, 0], clustered_img[:, :, 1]) / (2 * np.pi)) % 1.0
    new_sat = clustered_img[:, :, 2]
    new_val = clustered_img[:, :, 3]
    
    # Final HSV image (HxWx3)
    clustered_hsv = np.stack([new_hue, new_sat, new_val], axis=-1)
    
    return clustered_hsv


import os
import matplotlib.pyplot as plt
from skimage.io import imread
from cellpose import models, plot
from imageio import imwrite

# Your save_image_gt function
def save_image_gt(img, name, ext="tiff"):
    # Path to the Downloads folder (macOS)
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    filename = f"{name}.{ext.lstrip('.')}"
    output_path = os.path.join(downloads, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        # Save in 16-bit to preserve cell IDs
        imwrite(output_path, img.astype("uint16"))
        print(f"✅ Image saved at: {output_path}")
    except Exception as e:
        print(f"❌ Error saving: {output_path}")
        print(e)
