import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def robust_image_loader(path):
    """
    Lädt ein Bild, konvertiert falls nötig in RGB und liefert als np.ndarray zurück.
    Unterstützt TIFF, PNG, JPG.
    """
    from PIL import Image
    img = Image.open(path)
    img = img.convert('RGB')  # Sicherstellen, dass Bild 3 Kanäle hat
    return np.array(img)

def convert_to_grayscale(img):
    """
    Wandelt RGB- oder Graustufenbild in ein 2D Graustufenbild um.
    Falls schon Graustufen, wird nichts gemacht.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif len(img.shape) == 2:
        return img
    else:
        raise ValueError(f"Unerwartetes Bildformat: {img.shape}")

def normalize_image(img):
    """
    Skaliert Bildpixel auf 0-255 und Typ uint8.
    """
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)

def binarize_image(img, method='threshold', threshold=128):
    """
    Binarisiert ein Graustufenbild mit einem Schwellenwert.
    """
    if method == 'threshold':
        _, bin_img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    elif method == 'nonzero':
        bin_img = (img > 0).astype(np.uint8)
    else:
        raise ValueError(f"Unbekannte Binarisierungs-Methode: {method}")
    return bin_img

def dice_coefficient(y_true, y_pred):
    """
    Berechnet den Dice Koeffizienten zwischen zwei binären Masken.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    intersection = np.logical_and(y_true, y_pred).sum()
    size_sum = y_true.sum() + y_pred.sum()
    if size_sum == 0:
        return 1.0  # Wenn beide leer sind
    return 2.0 * intersection / size_sum

def evaluate_and_plot_dice_cells(image_pairs, title="Dice Score Vergleich"):
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
        plt.suptitle(f"Überprüfung: {os.path.basename(pred_path)}")

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
        plt.title("Überlagerung")

        plt.tight_layout()
        plt.show()

    # Ergebnisse als DataFrame und Barplot
    df = pd.DataFrame(results)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    n_bars = len(df)
    colors = sns.color_palette("hsv", n_bars)  # unterschiedliche Farben pro Balken

    barplot = sns.barplot(data=df, x='Bild', y='DiceScore', hue='Bild', palette=colors, edgecolor='black', dodge=False)
    plt.legend([],[], frameon=False)  # Legende ausblenden, wenn du keine Legende willst
    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.3f}',
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.ylabel("Dice Score")
    plt.xlabel("Bild")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return df

import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_segmentation_line(otsu_pairs, kmeans_pairs, title="Vergleich der Methoden (Dice Score)"):
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
    plt.xlabel("Bild (tXX)", fontsize=14, fontweight='bold')
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