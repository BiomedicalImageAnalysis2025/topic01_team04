import cv2 as cv2
import matplotlib.pyplot as plt
import os


def rgb_to_hsv(image):
    """
    Konvertiert ein RGB-Bild (numpy array) in den HSV-Farbraum.
    
    Parameter:
    image (numpy.ndarray): Eingabebild im RGB-Format
    
    Rückgabe:
    image_hsv (numpy.ndarray): Bild im HSV-Farbraum
    """
    if image is None:
        print("Fehler: Bild ist None.")
        return None

    # Konvertierung von RGB (matplotlib) zu HSV (OpenCV erwartet BGR, daher zuerst RGB->BGR)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return image_hsv

def hue_channel(hsv_image):
    """
    Extract the Hue channel from the HSV image.
    
    Parameters:
    hsv_image (numpy.ndarray): Input HSV image.
    
    Returns:
    numpy.ndarray: Hue channel.
    """
    return hsv_image[:, :, 0]


def saturation_channel(hsv_image):
    """
    Extract the Saturation channel from the HSV image.

    Parameters:
    hsv_image (numpy.ndarray): Input HSV image.
    Returns:
    numpy.ndarray: Saturation channel.
    """
    return hsv_image[:, :, 1]


def value_channel(hsv_image):
    """ Extract the Value channel from the HSV image.
    Parameters:
    hsv_image (numpy.ndarray): Input HSV image.
    Returns:
    numpy.ndarray: Value channel.
    """
    return hsv_image[:, :, 2]


def plot_hsv_channels(hsv_image):
    """
    Plot the Hue, Saturation, and Value channels of the HSV image.
    
    Parameters:
    hsv_image (numpy.ndarray): Input HSV image.
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(hue_channel(hsv_image), cmap='hsv')
    plt.axis('off')
    plt.title('Hue Channel')
    
    plt.subplot(1, 3, 2)
    plt.imshow(saturation_channel(hsv_image), cmap='gray')
    plt.axis('off')
    plt.title('Saturation Channel')
    
    plt.subplot(1, 3, 3)
    plt.imshow(value_channel(hsv_image), cmap='gray')
    plt.axis('off')
    plt.title('Value Channel')
    
    plt.show()
    return hsv_image, hue_channel(hsv_image), saturation_channel(hsv_image), value_channel(hsv_image)


def save_hsv_channels(hsv_image, output_prefix):
    """
    Save the Hue, Saturation, and Value channels as images.
    Parameters:
    hsv_image (numpy.ndarray): Input HSV image.
    output_prefix (str): Prefix for the output image files.
    """
    cv2.imwrite(f'{output_prefix}_hue_channel.jpg', hue_channel(hsv_image))
    cv2.imwrite(f'{output_prefix}_saturation_channel.jpg', saturation_channel(hsv_image))
    cv2.imwrite(f'{output_prefix}_value_channel.jpg', value_channel(hsv_image))
    cv2.imwrite(f'{output_prefix}_hsv_image.jpg', hsv_image)
    return hsv_image, hue_channel(hsv_image), saturation_channel(hsv_image), value_channel(hsv_image)


def display_images(original, name="Image"):
    """Zeige Originalbild in Jupyter an"""

    plt.figure(figsize=(6, 6))
    plt.imshow(original)
    plt.title(name)
    plt.axis('off')
    plt.show()


def save_image(img, name, ext="png"):
    """
    Speichert ein NumPy-Bildarray `img` in den Downloads-Ordner.
    
    - `name`: Basis-Name der Datei (ohne Extension)
    - `ext`: gewünschte Dateiendung, z.B. "png", "jpg", "tif" (default: "png")
    
    Matplotlib/Pillow erkennt aus der Extension automatisch das Format.
    """
    # 1) Pfad zum Download-Ordner (Windows, macOS, Linux)
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    # 2) kompletten Dateinamen zusammenbauen
    filename = f"{name}.{ext.lstrip('.')}"
    output_path = os.path.join(downloads, filename)
    # 3) sicherstellen, dass es den Ordner gibt
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 4) speichern
    plt.imsave(output_path, img)
    print(f"Image saved to: {output_path}")