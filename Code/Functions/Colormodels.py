import cv2 as cv2
import matplotlib.pyplot as plt
import os


def rgb_to_hsv(image):
    """
    Converts an RGB image (numpy array) to the HSV
    
    Parameters:
    image (numpy.ndarray):  Input RGB image (as a numpy array).
    
    Returns:
    image_hsv (numpy.ndarray): Converted HSV image
    """
    # convert RGB (matplotlib) to HSV (OpenCV expects BGR, so first convert RGB->BGR)
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
    """show original image"""

    plt.figure(figsize=(4, 4))
    plt.imshow(original)
    plt.title(name)
    plt.axis('off')
    plt.show()


def save_image(img, name, ext="png"):
    """
    Save an image to the Downloads folder with a specified name and extension.
    
    - `name`:  Name of the image file (without extension).
    - `ext`:   File extension (e.g., 'png', 'jpg', 'tiff'). Default is 'png'.
    
    Matplotlib/Pillow  is used to save the image, which supports various formats.
    """
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    filename = f"{name}.{ext.lstrip('.')}"
    output_path = os.path.join(downloads, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.imsave(output_path, img)
    print(f"Image saved to: {output_path}")


def plot_histogram(image, title="Histogram", xlabel="Pixel Intensity", ylabel="Frequency"):
    """
    Plot the histogram of the image.
    
    Parameters:
    image (numpy.ndarray): Input image.
    title (str): Title of the histogram plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()


def plot_hsv_histograms(hsv_image):
    """
    show the histograms of the Hue, Saturation, and Value channels of an HSV image.
    
    Parameters:
    - hsv_image: numpy.ndarray 
    """
    h, s, v = cv2.split(hsv_image)
    channels = [h, s, v]
    titles = ['Hue', 'Saturation', 'Value']
    colors = ['r', 'g', 'b']
    
    plt.figure(figsize=(15, 4))
    for i in range(3):
        hist = cv2.calcHist([channels[i]], [0], None, [256], [0, 256])
        plt.subplot(1, 3, i+1)
        plt.plot(hist, color=colors[i])
        plt.title(f"Histogram – {titles[i]}")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.xlim([0, 256])
        plt.grid(True)
    plt.tight_layout()
    plt.show()


def hsv_to_rgb(hsv_image):
    """
    Converts an HSV image (numpy array) to RGB.

    Parameters:
    hsv_image (numpy.ndarray): Input HSV image.

    Returns:
    numpy.ndarray: Converted RGB image.
    """
    # convert HSV to BGR (OpenCV expects BGR, so first convert HSV->BGR)
    image_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    # convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb