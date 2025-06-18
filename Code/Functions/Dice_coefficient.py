import cv2  
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

