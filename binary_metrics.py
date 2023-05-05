import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# - Specificity
def specificity(y_true, y_pred):
    # specificity = TN / ( TN + FP )
    # Get sum of intersection of predicted negatives with ground truth negatives for true negatives (TN)
    tn = K.sum((1-y_true) * (1-K.round(y_pred)))
    # Get sum of ground truth negatives for actual negatives (equal to TN + FP)
    tn_fp = K.sum(1-y_true)
    return tn / (tn_fp + K.epsilon())

# - Sensitivity
def sensitivity(y_true, y_pred):
    # sensitivity = TP / ( TP + FN )
    # Get sum of intersection of predicted positives with ground truth positives for true positives (TP)
    tp = K.sum(y_true * K.round(y_pred))
    # Get sum of ground truth positives for actual positives (equal to TP + FN)
    tp_fn = K.sum(y_true)
    return tp / (tp_fn + K.epsilon())

# - Dice coefficient index (*Dci*)
def dice_loss(y_true, y_pred, smooth=1e-6):
    # Flatten the ground truth and predicted masks
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # Compute the intersection and union
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)

    # Compute the Dice coefficient
    dice_coef = (2.0 * intersection + smooth) / (union + smooth)

    # Compute the Dice loss as 1 - Dice coefficient
    dice_loss = 1.0 - dice_coef

    return dice_loss

def combined_loss_wrapper(alpha=0.5, smooth=1e-5):
    def combined_loss(y_true, y_pred):
        nonlocal alpha
        
        # Calculate binary cross entropy
        bce = tf.losses.binary_crossentropy(y_true, y_pred)

        # Calculate Dice loss
        dl = dice_loss(y_true, y_pred)

        # Calculate combined loss
        combined = alpha * bce + (1-alpha) * dl

        return combined
    return combined_loss