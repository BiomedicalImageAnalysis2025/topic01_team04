import cv2  
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# === Dice-Koeffizient berechnen ===
def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_bin = (y_true > 0).astype(np.uint8)
    y_pred_bin = (y_pred > 0).astype(np.uint8)
    
    y_true_f = y_true_bin.flatten()
    y_pred_f = y_pred_bin.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# === Default-Mappings für Segmentierungen ===
def get_default_maps():
    return {
        "Otsu": {
            "Otsu_t13.tiff": "man_seg13.tif",
            "Otsu_t52.tiff": "man_seg52.tif",
            "Otsu_t75.tiff": "man_seg75.tif",
            "Otsu_t79.tiff": "man_seg79.tif",
        },
        "KMeans": {
            "KMeans_t13.tiff": "man_seg13.tif",
            "KMeans_t52.tiff": "man_seg52.tif",
            "KMeans_t75.tiff": "man_seg75.tif",
            "KMeans_t79.tiff": "man_seg79.tif",
        }
    }

# === Auswertung von Segmentierungen mit Dice-Score ===
def evaluate_segmentation(base_dir: str,
                          method_paths: dict,
                          mappings: dict = None) -> pd.DataFrame:
    
    gt_folder = os.path.join(base_dir, "Original_Images", "Otsu", "Data", "N2DL-HeLa", "gt")
    results = []

    for method, folder in method_paths.items():
        file_map = mappings.get(method, {})
        for seg_name, gt_name in file_map.items():
            seg_path = os.path.join(folder, seg_name)
            gt_path = os.path.join(gt_folder, gt_name)

            seg_img = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            
            gt_norm = cv2.normalize(gt_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            gt_bin = (gt_norm > 0).astype(np.uint8)
            seg_bin = (seg_img > 0).astype(np.uint8)

            score = dice_coefficient(gt_bin, seg_bin)
            results.append({
                'Methode': method,
                'Datei': seg_name,
                'DiceScore': float(score)
            })

    return pd.DataFrame(results)

# === Barplot für Dice-Scores ===
def plot_dice_scores(df: pd.DataFrame,
                     x: str = 'Datei',
                     y: str = 'DiceScore',
                     hue: str = 'Methode') -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df, x=x, y=y, hue=hue, edgecolor="black", linewidth=1.2)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_ylim(0, 1)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Dice Score per Image and Method")
    plt.tight_layout()
    plt.show()


import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def best_cluster_mask(pred_img, gt_bin):
    """
    Findet die Kombination aller Cluster (außer Hintergrund 0) in pred_img,
    die den besten Dice Score gegenüber gt_bin ergibt.
    Prüft auch die invertierte Maske.
    """
    cluster_ids = np.unique(pred_img)
    
    # Erzeuge Maske mit allen Clustern außer 0 (Hintergrund)
    combined_mask = np.zeros_like(gt_bin, dtype=np.uint8)
    for cid in cluster_ids:
        if cid == 0:
            continue
        combined_mask = np.logical_or(combined_mask, pred_img == cid)
    combined_mask = combined_mask.astype(np.uint8)
    
    # Invertierte Maske erstellen
    inverted_mask = 1 - combined_mask
    
    # Dice Score berechnen
    score_normal = dice_coefficient(gt_bin, combined_mask)
    score_inverted = dice_coefficient(gt_bin, inverted_mask)
    
    if score_inverted > score_normal:
        return inverted_mask, score_inverted
    else:
        return combined_mask, score_normal



def load_and_preprocess_image(path):
    """
    Lädt ein Bild, wandelt RGB zu Graustufen um, falls nötig.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Bild konnte nicht geladen werden: {path}")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def binarize_image(img, method='threshold', threshold=128):
    """
    Binarisiert das Bild:
    - 'threshold': klassischer Schwellenwert (>= threshold = 1)
    - 'nonzero': alles > 0 wird 1
    """
    if method == 'threshold':
        bin_img = (img >= threshold).astype(np.uint8)
    elif method == 'nonzero':
        bin_img = (img > 0).astype(np.uint8)
    else:
        raise ValueError(f"Unbekannte Binarisierungs-Methode: {method}")
    return bin_img


    return df

import pandas as pd
def evaluate_and_plot_dice(image_pairs, title="Dice Score Vergleich"):
    results = []

    for pred_path, gt_path in image_pairs:
        try:
            y_pred = load_and_preprocess_image(pred_path)
            y_true = load_and_preprocess_image(gt_path)
        except FileNotFoundError as e:
            # Optional: Fehler ignorieren oder loggen
            continue

        # Shapes angleichen, wenn nötig
        if y_pred.shape != y_true.shape:
            y_true = cv2.resize(y_true, (y_pred.shape[1], y_pred.shape[0]), interpolation=cv2.INTER_NEAREST)

        y_true_bin = binarize_image(y_true, method='nonzero')

        unique_pred = np.unique(y_pred)

        if len(unique_pred) > 2:
            y_pred_mask, dice_score = best_cluster_mask(y_pred, y_true_bin)

            inverted_mask = 1 - y_pred_mask
            dice_inverted = dice_coefficient(y_true_bin, inverted_mask)

            if dice_inverted > dice_score:
                y_pred_bin = inverted_mask
                dice_score = dice_inverted
            else:
                y_pred_bin = y_pred_mask

        else:
            y_pred_bin = binarize_image(y_pred, method='threshold', threshold=128)

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
        plt.imshow(y_true_bin, cmap='Reds', alpha=0.5)
        plt.imshow(y_pred_bin, cmap='Blues', alpha=0.5)
        plt.title("Überlagerung")

        plt.tight_layout()
        plt.show()

    df = pd.DataFrame(results)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(data=df, x='Bild', y='DiceScore', color='skyblue', edgecolor='black')
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

#Same as evaluate_and_plot_dice but with colored overlay
def evaluate_and_plot_dice_colored(image_pairs, title="Dice Score Vergleich"):
    results = []

    for pred_path, gt_path in image_pairs:
        try:
            y_pred = load_and_preprocess_image(pred_path)
            y_true = load_and_preprocess_image(gt_path)
        except FileNotFoundError as e:
            # Optional: Fehler ignorieren oder loggen
            continue

        # Shapes angleichen, wenn nötig
        if y_pred.shape != y_true.shape:
            y_true = cv2.resize(y_true, (y_pred.shape[1], y_pred.shape[0]), interpolation=cv2.INTER_NEAREST)

        y_true_bin = binarize_image(y_true, method='nonzero')

        unique_pred = np.unique(y_pred)

        if len(unique_pred) > 2:
            y_pred_mask, dice_score = best_cluster_mask(y_pred, y_true_bin)

            inverted_mask = 1 - y_pred_mask
            dice_inverted = dice_coefficient(y_true_bin, inverted_mask)

            if dice_inverted > dice_score:
                y_pred_bin = inverted_mask
                dice_score = dice_inverted
            else:
                y_pred_bin = y_pred_mask

        else:
            y_pred_bin = binarize_image(y_pred, method='threshold', threshold=128)

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
        plt.imshow(y_true_bin, cmap='Reds', alpha=0.5)
        plt.imshow(y_pred_bin, cmap='Blues', alpha=0.5)
        plt.title("Überlagerung")

        plt.tight_layout()
        plt.show()

    df = pd.DataFrame(results)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    n_bars = len(df)
    colors = sns.color_palette("hsv", n_bars)  # unterschiedliche Farben pro Balken

    barplot = sns.barplot(data=df, x='Bild', y='DiceScore', palette=colors, edgecolor='black')
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

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd
import seaborn as sns

def robust_image_loader(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext in ['.tif', '.tiff']:
        img = Image.open(path)
        img = np.array(img)
    else:
        img = plt.imread(path)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]
    return img



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image

def evaluate_segmentation_spaghetti(otsu_pairs, kmeans_pairs, title="Vergleich der Methoden (Dice Score)"):

    def robust_image_loader(path):
        img = Image.open(path)
        img = img.convert('RGB')
        return np.array(img)

    def convert_to_grayscale(img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif len(img.shape) == 2:
            return img
        else:
            raise ValueError(f"Unerwartetes Bildformat: {img.shape}")

    def normalize_image(img):
        img = img.astype(np.float32)
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255
        else:
            img = np.zeros_like(img)
        return img.astype(np.uint8)

    def binarize_image(img, method='threshold', threshold=128):
        if method == 'threshold':
            _, bin_img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
        elif method == 'nonzero':
            bin_img = (img > 0).astype(np.uint8)
        else:
            raise ValueError(f"Unbekannte Binarisierungs-Methode: {method}")
        return bin_img

    def dice_coefficient(y_true, y_pred):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        intersection = np.logical_and(y_true, y_pred).sum()
        size_sum = y_true.sum() + y_pred.sum()
        if size_sum == 0:
            return 1.0
        return 2.0 * intersection / size_sum

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
                'DiceScore': dice_score,
                'Methode': methode_name
            })

        return pd.DataFrame(results)

    # --- Auswertung für beide Methoden ---
    df_otsu = evaluate(otsu_pairs, methode_name="Otsu")
    df_kmeans = evaluate(kmeans_pairs, methode_name="KMeans")

    df_combined = pd.concat([df_otsu, df_kmeans], ignore_index=True)

    # --- Spaghetti-Plot ---
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")

    method_order = ["Otsu", "KMeans"]  # feste Reihenfolge auf x-Achse
    x_positions = {methode: i for i, methode in enumerate(method_order)}  # numerische Positionen

    for bild, gruppe in df_combined.groupby('Bild'):
        x_vals = [x_positions[m] for m in gruppe['Methode']]
        y_vals = gruppe['DiceScore'].values

        # Linie zwischen den beiden Punkten desselben Bildes
        plt.plot(x_vals, y_vals, marker='o', linewidth=2, label=bild)

        # Bildnamen leicht oberhalb der Punkte anzeigen
        for x, y in zip(x_vals, y_vals):
            plt.text(x, y + 0.02, bild, ha='center', fontsize=8, rotation=30)

    plt.xticks(ticks=list(x_positions.values()), labels=method_order)
    plt.ylim(0.4, 1.05)
    plt.xlabel("Methode")
    plt.ylabel("Dice Score")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return df_combined

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import re

def evaluate_segmentation_line(otsu_pairs, kmeans_pairs, title="Vergleich der Methoden (Dice Score)"):

    def robust_image_loader(path):
        img = Image.open(path)
        img = img.convert('RGB')
        return np.array(img)

    def convert_to_grayscale(img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif len(img.shape) == 2:
            return img
        else:
            raise ValueError(f"Unerwartetes Bildformat: {img.shape}")

    def normalize_image(img):
        img = img.astype(np.float32)
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255
        else:
            img = np.zeros_like(img)
        return img.astype(np.uint8)

    def binarize_image(img, method='threshold', threshold=128):
        if method == 'threshold':
            _, bin_img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
        elif method == 'nonzero':
            bin_img = (img > 0).astype(np.uint8)
        else:
            raise ValueError(f"Unbekannte Binarisierungs-Methode: {method}")
        return bin_img

    def dice_coefficient(y_true, y_pred):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        intersection = np.logical_and(y_true, y_pred).sum()
        size_sum = y_true.sum() + y_pred.sum()
        if size_sum == 0:
            return 1.0
        return 2.0 * intersection / size_sum

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
    df_combined['TName'] = pd.Categorical(df_combined['TName'], 
                                          categories=sorted(df_combined['TName'].unique(), key=lambda x: int(x[1:])),
                                          ordered=True)

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



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import re

def evaluate_segmentation_line_test(otsu_pairs, kmeans_pairs, title="Vergleich der Methoden (Dice Score)"):

    def robust_image_loader(path):
        img = Image.open(path)
        img = img.convert('RGB')
        return np.array(img)

    def convert_to_grayscale(img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif len(img.shape) == 2:
            return img
        else:
            raise ValueError(f"Unerwartetes Bildformat: {img.shape}")

    def normalize_image(img):
        img = img.astype(np.float32)
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255
        else:
            img = np.zeros_like(img)
        return img.astype(np.uint8)

    def binarize_image(img, method='threshold', threshold=128):
        if method == 'threshold':
            _, bin_img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
        elif method == 'nonzero':
            bin_img = (img > 0).astype(np.uint8)
        else:
            raise ValueError(f"Unbekannte Binarisierungs-Methode: {method}")
        return bin_img

    def dice_coefficient(y_true, y_pred):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        intersection = np.logical_and(y_true, y_pred).sum()
        size_sum = y_true.sum() + y_pred.sum()
        if size_sum == 0:
            return 1.0
        return 2.0 * intersection / size_sum

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
    df_combined['TName'] = pd.Categorical(df_combined['TName'], 
                                          categories=sorted(df_combined['TName'].unique(), key=lambda x: int(x[1:])),
                                          ordered=True)

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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def evaluate_dice_all_in_one(image_pairs, title="Dice Score Vergleich"):
    import os
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from PIL import Image

    def robust_image_loader(path):
        img = Image.open(path)
        img = img.convert('RGB')
        return np.array(img)

    def convert_to_grayscale(img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif len(img.shape) == 2:
            return img
        else:
            raise ValueError(f"Unerwartetes Bildformat: {img.shape}")

    def normalize_image(img):
        img = img.astype(np.float32)
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255
        else:
            img = np.zeros_like(img)
        return img.astype(np.uint8)

    def binarize_image(img, method='threshold', threshold=128):
        if method == 'threshold':
            _, bin_img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
        elif method == 'nonzero':
            bin_img = (img > 0).astype(np.uint8)
        else:
            raise ValueError(f"Unbekannte Binarisierungs-Methode: {method}")
        return bin_img

    def dice_coefficient(y_true, y_pred):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        intersection = np.logical_and(y_true, y_pred).sum()
        size_sum = y_true.sum() + y_pred.sum()
        if size_sum == 0:
            return 1.0
        return 2.0 * intersection / size_sum

    def is_binary(img):
        values = np.unique(img)
        return set(values).issubset({0, 1}) or set(values).issubset({0, 255}) or len(values) <= 2

    results = []

    for pred_path, gt_path in image_pairs:
        print(f"\n--- Vergleiche: {os.path.basename(pred_path)} mit {os.path.basename(gt_path)} ---")

        try:
            y_pred = robust_image_loader(pred_path)
            y_true = robust_image_loader(gt_path)
        except FileNotFoundError as e:
            print(f"Datei nicht gefunden: {e}")
            continue

        y_pred_gray = convert_to_grayscale(y_pred)
        y_true_gray = convert_to_grayscale(y_true)

        print("Unique Werte vor Verarbeitung:")
        print("Prediction:", np.unique(y_pred_gray))
        print("Ground Truth:", np.unique(y_true_gray))

        if not is_binary(y_pred_gray):
            y_pred_gray = normalize_image(y_pred_gray)

        unique_gt = np.unique(y_true_gray)
        if len(unique_gt) > 20:
            print(f"⚠️ Achtung: Ground Truth enthält viele Klassen ({len(unique_gt)}) — wird als binäre Maske vereinfacht.")

        y_true_bin = binarize_image(y_true_gray, method='nonzero')

        if is_binary(y_pred_gray):
            y_pred_bin = (y_pred_gray > 0).astype(np.uint8)
            print("Prediction war bereits binär.")
        else:
            y_pred_bin = binarize_image(y_pred_gray, method='threshold', threshold=128)
            print("Prediction wurde binarisiert mit Schwelle 128.")

        dice_score = dice_coefficient(y_true_bin, y_pred_bin)
        dice_inverted = dice_coefficient(y_true_bin, 1 - y_pred_bin)
        if dice_inverted > dice_score:
            print("Maske invertiert, weil Score besser.")
            y_pred_bin = 1 - y_pred_bin
            dice_score = dice_inverted

        print("Unique Werte nach Verarbeitung:")
        print("Prediction:", np.unique(y_pred_bin))
        print("Ground Truth:", np.unique(y_true_bin))
        print(f"Dice Score: {dice_score:.4f}")

        results.append({'Bild': os.path.splitext(os.path.basename(pred_path))[0], 'DiceScore': dice_score})

        # Debug-Visualisierung
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Überprüfung: {os.path.basename(pred_path)}")

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
        plt.title("Überlagerung")

        plt.tight_layout()
        plt.show()

    # Balkendiagramm
    df = pd.DataFrame(results)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("hsv", len(df))
    barplot = sns.barplot(data=df, x='Bild', y='DiceScore', hue='Bild', palette=colors, edgecolor='black', dodge=False)
    plt.legend([], [], frameon=False)
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


def evaluate_and_plot_dice_cells_normal_color(image_pairs, title="Dice Score Vergleich"):
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
    colors = sns.color_palette("pastel", n_bars)  # unterschiedliche Farben pro Balken

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

import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_and_plot_dice_cells_test1(image_pairs, title="Dice Score Vergleich"):
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
        dice_inverted = dice_coefficient(y_true_bin, 1 - y_pred_bin)

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
        plt.suptitle(f"Überprüfung: {raw_name}")

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
        plt.title("Überlagerung")

        plt.tight_layout()
        plt.show()

    # DataFrame + Barplot
    df = pd.DataFrame(results)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    n_bars = len(df)
    colors = sns.color_palette("Paired", n_bars)

    barplot = sns.barplot(data=df, x='Bild', y='DiceScore', hue='Bild', palette=colors, edgecolor='black', dodge=False)
    plt.legend([], [], frameon=False)  # Keine Legende

    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.3f}',
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=0, ha='center', fontsize=10)  # Waagrecht, keine Rotation
    plt.ylim(0, 1)
    plt.ylabel("Dice Score")
    plt.xlabel("Bild")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return df

def evaluate_and_plot_dice_cells_test2(image_pairs, title="Dice Score Comparison"):
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

        # Nur der normale Dice Score (kein Invertieren)
        dice_score = dice_coefficient(y_true_bin, y_pred_bin)

        raw_name = os.path.splitext(os.path.basename(pred_path))[0]
        clean_name = raw_name
        clean_name = clean_name.replace("Prediction_", "")
        clean_name = clean_name.replace("norm", "")
        clean_name = clean_name.replace("clustered", "kmeans")
        clean_name = clean_name.replace("seg", "GT")
        clean_name = clean_name.replace("_", "\n")

        results.append({'Bild': clean_name, 'DiceScore': dice_score})

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

    df = pd.DataFrame(results)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    n_bars = len(df)
    colors = sns.color_palette("Paired", n_bars)

    barplot = sns.barplot(data=df, x='Bild', y='DiceScore', hue='Bild', palette=colors, edgecolor='black', dodge=False)
    plt.legend([], [], frameon=False)

    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.3f}',
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=0, ha='center', fontsize=10)
    plt.ylim(0, 1)
    plt.ylabel("Dice Score")
    plt.xlabel("Picture")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return df
