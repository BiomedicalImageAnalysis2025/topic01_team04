import cv2  
import numpy as np

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """
    Compute the Dice coefficient between two binary masks.
    
    Args:
        y_true (np.ndarray): Ground truth binary mask.
        y_pred (np.ndarray): Predicted binary mask.
        smooth (float): Smoothing factor to avoid division by zero.
        
    Returns:
        float: Dice coefficient value.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


