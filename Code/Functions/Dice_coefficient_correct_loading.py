# This code loads image pairs (prediction and ground truth), calculates the Dice score for segmentation evaluation and 
# visualizes the results   

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load an image robustly from various formats and convert it to RGB if necessary
def robust_image_loader(path):
    """
    Loads an image, converts it to RGB if needed, and returns it as a NumPy array.
    Supports TIFF, PNG, JPG formats.
    """
    from PIL import Image
    img = Image.open(path)
    img = img.convert('RGB')  
    return np.array(img)

# Convert an image to grayscale if it's RGB; otherwise, return as-is
def convert_to_grayscale(img):
    """
    Converts RGB or grayscale image into a 2D grayscale image.
    If already grayscale, does nothing.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif len(img.shape) == 2:
        return img
    else:
        raise ValueError(f"Unerwartetes Bildformat: {img.shape}")
    
# Normalize image values to the 0–255 range and ensure type is uint8
def normalize_image(img):
    """
    Scales image pixels to range 0–255 and type uint8.
    """
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)

# Binarize image using a threshold or by checking for non-zero pixels
def binarize_image(img, method='threshold', threshold=128):
    """
    Binarizes a grayscale image using a threshold.
    """
    if method == 'threshold':
        _, bin_img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    elif method == 'nonzero':
        bin_img = (img > 0).astype(np.uint8)
    else:
        raise ValueError(f"Unbekannte Binarisierungs-Methode: {method}")
    return bin_img

# Calculate the Dice coefficient between two binary masks
def dice_coefficient(y_true, y_pred):
    """
    Computes the Dice coefficient between two binary masks
    """
    #Is a pixel segment (True) or not a segment (False)
    #bool was used to ensure that the binary masks are interpreted as true logical values, so that logical operations such as overlap (e.g., for the Dice Score) work correctly and reliably.
    y_true = y_true.astype(bool) # Convert to boolean for logical operations
    y_pred = y_pred.astype(bool)

    intersection = np.logical_and(y_true, y_pred).sum() # Count overlapping pixels
    size_sum = y_true.sum() + y_pred.sum() # Total number of positive pixels
    #If both masks show nothing, they did the same thing – so the dice score is 1.0
    if size_sum == 0:
        return 1.0  # If both are empty, consider perfect match
    return 2.0 * intersection / size_sum ## Calculates Dice coefficient: 2 × overlap divided by total size of both masks

# Evaluate and visualize Dice coefficient for multiple image pairs
def evaluate_and_plot_dice_cells(image_pairs, title="Dice Score Comparison"):
    results = [] ## Initialize list to store results

    for pred_path, gt_path in image_pairs: # Loop over predicted and ground truth paths
        try:
            y_pred = robust_image_loader(pred_path)
            y_true = robust_image_loader(gt_path)
        except FileNotFoundError as e:
            print(f"Datei nicht gefunden: {e}")
            continue

        # Beide Bilder auf gleichen Farbraum (Graustufen)
        y_pred_gray = convert_to_grayscale(y_pred)
        y_true_gray = convert_to_grayscale(y_true)

        # Beide Bilder normalisieren (0-255 uint8)
        y_pred_norm = normalize_image(y_pred_gray)
        y_true_norm = normalize_image(y_true_gray)

        # Größenausgleich
        if y_pred_norm.shape != y_true_norm.shape:
            y_true_norm = cv2.resize(y_true_norm, (y_pred_norm.shape[1], y_pred_norm.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Binarisierung der GT (nonzero)
        y_true_bin = binarize_image(y_true_norm, method='nonzero')

        # Binarisierung der Prediction (threshold)
        y_pred_bin = binarize_image(y_pred_norm, method='threshold', threshold=128)

        # Dice Score berechnen und ggf. Maske invertieren, falls besser
        dice_score = dice_coefficient(y_true_bin, y_pred_bin)
        dice_inverted = dice_coefficient(y_true_bin, 1 - y_pred_bin)

        if dice_inverted > dice_score:
            y_pred_bin = 1 - y_pred_bin
            dice_score = dice_inverted

        results.append({'Bild': os.path.basename(pred_path), 'DiceScore': dice_score})

        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Overlap of segmented Images: {os.path.basename(pred_path)}")

        plt.subplot(1, 3, 1)
        plt.imshow(y_true_bin, cmap='gray')
        plt.title("Ground Truth")

        plt.subplot(1, 3, 2)
        plt.imshow(y_pred_bin, cmap='gray')
        plt.title("Prediction")

        plt.subplot(1, 3, 3)
        # Overlay: Ground Truth Rot, Prediction Blau
        overlay = np.zeros((*y_true_bin.shape, 3), dtype=np.uint8)
        overlay[y_true_bin == 1] = [255, 0, 0]     # Rot GT
        overlay[y_pred_bin == 1] = [0, 0, 255]     # Blau Prediction
        plt.imshow(overlay)
        plt.title("Overlap")

        plt.tight_layout()
        plt.show()

    # Ergebnisse als DataFrame und Barplot
    df = pd.DataFrame(results)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    n_bars = len(df)
    colors = sns.color_palette("tab10", n_bars)  # unterschiedliche Farben pro Balken

    barplot = sns.barplot(data=df, x='Bild', y='DiceScore', hue='Bild', palette=colors, edgecolor='black', dodge=False)
    plt.legend([],[], frameon=False)  # Legende ausblenden, wenn du keine Legende willst
    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.3f}',
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.ylabel("Dice Score")
    plt.xlabel("Picture")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return df

import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_segmentation_line(otsu_pairs, kmeans_pairs, title="Comparison of methods (Dice Score)"):
    def extract_tname(filename):
        match = re.search(r't\d+', filename)
        return match.group(0) if match else filename

    def evaluate(image_pairs, methode_name):
        results = []

        for pred_path, gt_path in image_pairs:
            try:
                y_pred = robust_image_loader(pred_path)
                y_true = robust_image_loader(gt_path)
            except FileNotFoundError as e:
                print(f"Datei nicht gefunden: {e}")
                continue

            y_pred_gray = convert_to_grayscale(y_pred)
            y_true_gray = convert_to_grayscale(y_true)

            y_pred_norm = normalize_image(y_pred_gray)
            y_true_norm = normalize_image(y_true_gray)

            if y_pred_norm.shape != y_true_norm.shape:
                y_true_norm = cv2.resize(y_true_norm, (y_pred_norm.shape[1], y_pred_norm.shape[0]), interpolation=cv2.INTER_NEAREST)

            y_true_bin = binarize_image(y_true_norm, method='nonzero')
            y_pred_bin = binarize_image(y_pred_norm, method='threshold', threshold=128)

            dice_score = dice_coefficient(y_true_bin, y_pred_bin)
            dice_inverted = dice_coefficient(y_true_bin, 1 - y_pred_bin)

            if dice_inverted > dice_score:
                dice_score = dice_inverted

            results.append({
                'Bild': os.path.basename(pred_path),
                'TName': extract_tname(pred_path),
                'DiceScore': dice_score,
                'Methode': methode_name
            })

        return pd.DataFrame(results)

    # Auswertung
    df_otsu = evaluate(otsu_pairs, methode_name="Otsu")
    df_kmeans = evaluate(kmeans_pairs, methode_name="KMeans")
    df_combined = pd.concat([df_otsu, df_kmeans], ignore_index=True)

    # Sortierung nach TName
    df_combined['TName'] = pd.Categorical(
        df_combined['TName'],
        categories=sorted(df_combined['TName'].unique(), key=lambda x: int(x[1:])),
        ordered=True
    )

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    sns.lineplot(data=df_combined, x="TName", y="DiceScore", hue="Methode", marker="o", linewidth=2)

    plt.ylim(0.4, 1.05)
    plt.xlabel("Picture (tXX)", fontsize=14, fontweight='bold')
    plt.ylabel("Dice Score", fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')

    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    plt.legend(
        title="Methode",
        title_fontsize=18,
        fontsize=13,
        loc="upper right"
    )

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_and_plot_dice_cells_filter_otsu_kmeans(image_pairs, title="Dice Score Comparison"):
    results = []

    for pred_path, gt_path in image_pairs:
        try:
            y_pred = robust_image_loader(pred_path)
            y_true = robust_image_loader(gt_path)
        except FileNotFoundError as e:
            print(f"Datei nicht gefunden: {e}")
            continue

        # Beide Bilder auf gleichen Farbraum (Graustufen)
        y_pred_gray = convert_to_grayscale(y_pred)
        y_true_gray = convert_to_grayscale(y_true)

        # Normalisieren (0–255, uint8)
        y_pred_norm = normalize_image(y_pred_gray)
        y_true_norm = normalize_image(y_true_gray)

        # Größenangleichung
        if y_pred_norm.shape != y_true_norm.shape:
            y_true_norm = cv2.resize(y_true_norm, (y_pred_norm.shape[1], y_pred_norm.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Binarisierung
        y_true_bin = binarize_image(y_true_norm, method='nonzero')
        y_pred_bin = binarize_image(y_pred_norm, method='threshold', threshold=128)

        # Dice Score (ggf. invertiert)
        dice_score = dice_coefficient(y_true_bin, y_pred_bin)
        dice_inverted = dice_coefficient(y_true_bin, 1 - y_pred_bin) # Check if inverted prediction gives better score

        if dice_inverted > dice_score:
            y_pred_bin = 1 - y_pred_bin
            dice_score = dice_inverted

        # Lesbarer Bildname: Otsu-Bilder behalten 'Otsu_' im Namen
        raw_name = os.path.splitext(os.path.basename(pred_path))[0]
        # Entferne nur "Prediction_", "norm" und ersetze "seg" bei Ground Truth
        clean_name = raw_name
        clean_name = clean_name.replace("Prediction_", "")
        clean_name = clean_name.replace("norm", "")
        clean_name = clean_name.replace("clustered", "kmeans")
        clean_name = clean_name.replace("seg", "GT")
        clean_name = clean_name.replace("_", "\n")

        results.append({'Bild': clean_name, 'DiceScore': dice_score})

        # Vergleichsplots anzeigen
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Overlap of segmented Images: {raw_name}")

        plt.subplot(1, 3, 1)
        plt.imshow(y_true_bin, cmap='gray')
        plt.title("Ground Truth")

        plt.subplot(1, 3, 2)
        plt.imshow(y_pred_bin, cmap='gray')
        plt.title("Prediction")

        plt.subplot(1, 3, 3)
        overlay = np.zeros((*y_true_bin.shape, 3), dtype=np.uint8)
        overlay[y_true_bin == 1] = [255, 0, 0]
        overlay[y_pred_bin == 1] = [0, 0, 255]
        plt.imshow(overlay)
        plt.title("Overlap")

        plt.tight_layout()
        plt.show()

    # DataFrame + Barplot
    df = pd.DataFrame(results) # Convert results to DataFrame

    sns.set_theme(style="whitegrid") # Set theme for seaborn
    plt.figure(figsize=(10, 6)) #Crate figure

    n_bars = len(df)
    colors = sns.color_palette("Paired", n_bars)

    barplot = sns.barplot(data=df, x='Bild', y='DiceScore', hue='Bild', palette=colors, edgecolor='black', dodge=False)
    plt.legend([], [], frameon=False)  # Hide Legend

    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.3f}',
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=0, ha='center', fontsize=10)  # Rotate x-labels
    plt.ylim(0, 1)
    plt.ylabel("Dice Score")
    plt.xlabel("Picture")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return df
