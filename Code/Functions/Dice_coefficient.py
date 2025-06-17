import cv2  
import numpy as np
import os
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """
    Compute the Dice coefficient between two binary masks.
    
    Args:
        y_true (np.ndarray): Ground truth mask (can be grayscale or binary).
        y_pred (np.ndarray): Predicted mask (can be grayscale or binary).
        smooth (float): Smoothing factor to avoid division by zero.
        
    Returns:
        float: Dice coefficient value.
    """
    # Binarize inputs: alles größer 0 wird zu 1, sonst 0
    y_true_bin = (y_true > 0).astype(np.uint8)
    y_pred_bin = (y_pred > 0).astype(np.uint8)
    
    y_true_f = y_true_bin.flatten()
    y_pred_f = y_pred_bin.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)






# Default mappings for segmentation methods - Which segmentation file belongs to which ground truth file?
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

def evaluate_segmentation(base_dir: str,
                          method_paths: dict,
                          mappings: dict = None) -> pd.DataFrame:
    """
    Compute Dice scores for specified segmentation methods.

    Args:
        base_dir (str): Base directory of the project.
        method_paths (dict): Dict of method names to their segmentation folder paths.
        mappings (dict, optional): Dict of method-specific filename mappings. Uses get_default_maps if None.

    Returns:
        pd.DataFrame: DataFrame with columns ['Methode', 'Datei', 'DiceScore']
    """
    

    # Ground truth folder is fixed - it was already given
    gt_folder = os.path.join(base_dir, "Original_Images", "Otsu", "Data", "N2DL-HeLa", "gt")
    results = []

    for method, folder in method_paths.items():
        file_map = mappings.get(method, {})
        for seg_name, gt_name in file_map.items(): #Go through each segmentation image and the corresponding ground truth
            seg_path = os.path.join(folder, seg_name)
            gt_path = os.path.join(gt_folder, gt_name)

            seg_img = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            

            # Normalize and binarize ground truth (all non-zero as mask)
            gt_norm = cv2.normalize(gt_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            gt_bin = (gt_norm > 0).astype(np.uint8)

            # Binarize segmentation
            seg_bin = (seg_img > 0).astype(np.uint8)

            # Compute Dice
            score = dice_coefficient(gt_bin, seg_bin)
            results.append({
                'Methode': method,
                'Datei': seg_name,
                'DiceScore': float(score)
            })

    return pd.DataFrame(results)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
import os

# === Funktion zum Plotten der Dice Scores ===
def plot_dice_scores(df: pd.DataFrame,
                     x: str = 'Datei',
                     y: str = 'DiceScore',
                     hue: str = 'Methode') -> None:
    """
    Plot Dice scores in a barplot.

    Args:
        df (pd.DataFrame): DataFrame from evaluate_segmentation.
        x (str): Column name for x-axis.
        y (str): Column name for y-axis.
        hue (str): Column name for hue grouping.
    """
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


# Functions/evaluate_dice.py

def evaluate_and_plot_dice(image_pairs, title="Dice Score Vergleich"):
    sns.set_theme(style="whitegrid")
    results = []

    for pred_path, gt_path in image_pairs:
        y_pred = np.copy(plt.imread(pred_path))
        y_true = np.copy(plt.imread(gt_path))

        if y_pred.ndim == 3:
            y_pred = np.dot(y_pred[..., :3], [0.2989, 0.5870, 0.1140])
        if y_true.ndim == 3:
            y_true = np.dot(y_true[..., :3], [0.2989, 0.5870, 0.1140])

        y_pred_bin = (y_pred > 0).astype(np.uint8)
        y_true_bin = (y_true > 0).astype(np.uint8)

        if y_pred_bin.shape != y_true_bin.shape:
            y_true_bin = cv2.resize(y_true_bin, (y_pred_bin.shape[1], y_pred_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

        score = dice_coefficient(y_true_bin, y_pred_bin)
        short_name = os.path.basename(pred_path)
        results.append({'Bild': short_name, 'DiceScore': score})

    df = pd.DataFrame(results)
    print(df)

    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(data=df, x='Bild', y='DiceScore', width=0.4, color='skyblue')

    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.2f}',
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=30, ha='right')
    plt.ylim(0, 1)
    plt.ylabel("Dice Score", fontsize=12)
    plt.xlabel("Bild", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

    return df


