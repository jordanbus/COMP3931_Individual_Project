import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def binary_cross_entropy_per_channel_wrapper(class_weights):
    def binary_cross_entropy_per_channel(y_true, y_pred):
        nonlocal class_weights
        channels = y_pred.shape[-1]
        
        # Use equal class weights if not defined or wrong length
        if class_weights is None or len(class_weights) != channels:
            class_weights = np.ones(channels)
            
        # Normalize class weights
        class_weights /= np.sum(class_weights)
        
        # Compute binary cross-entropy loss for each channel separately, with class weights
        loss = 0.0
        for i in range(channels):
            weighted_loss = class_weights[i] * tf.losses.binary_crossentropy(y_true[..., i], y_pred[..., i])
            loss += weighted_loss
        return loss
    return binary_cross_entropy_per_channel
    
def dice_loss_wrapper(class_weights, smooth=1e-6):
    def dice_loss(y_true, y_pred):
        nonlocal class_weights
        
        channels = y_pred.shape[-1]
        # Use equal class weights if not defined or wrong length
        if class_weights is None or len(class_weights) != channels:
            class_weights = np.ones(channels)
        
        # Normalize the class weights to sum to 1
        class_weights /= np.sum(class_weights)

        loss = 0
        for i in range(channels):
            loss += class_weights[i] *  _dice_loss_for_label(y_true, y_pred, i, smooth)
        return loss
    return dice_loss

def specificity(y_true, y_pred, smooth=1e-6):
    # specificity = TN / ( TN + FP )
    
    # Get sum of intersection of predicted negatives with ground truth negatives for true negatives (TN)
    # Don't include background in calculation
    tn = K.sum((1-y_true[...,1:]) * (1-K.round(y_pred[...,1:])))
    # Get sum of ground truth negatives for actual negatives (equal to TN + FP)
    tn_fp = K.sum(1-y_true[...,1:])
    return tn / (tn_fp + smooth)

# - Sensitivity
def sensitivity(y_true, y_pred, smooth=1e-6):
    # sensitivity = TP / ( TP + FN )
    
    # Get sum of intersection of predicted positives with ground truth positives for true positives (TP)
    # Don't include background in calculation
    tp = K.sum(y_true[...,1:] * K.round(y_pred[...,1:]))
    # Get sum of ground truth positives for actual positives (equal to TP + FN)
    tp_fn = K.sum(y_true[...,1:])
    return tp / (tp_fn + smooth)

# - Dice coefficient index (*Dci*)
def _dice_loss_for_label(y_true, y_pred, label, smooth=1e-6):
    y_true_f = K.flatten(y_true[...,label])
    y_pred_f = K.flatten(y_pred[...,label])

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)

    dice_coef = (2.0 * intersection + smooth) / (union + smooth)

    # Compute the Dice loss as 1 - Dice coefficient
    dice_loss = 1.0 - dice_coef

    return dice_loss

# - Dice coefficient index for each class
def dice_coef_necrotic(y_true, y_pred, smooth=1e-6):
    return 1 - _dice_loss_for_label(y_true, y_pred, 1, smooth)

def dice_coef_edema(y_true, y_pred, smooth=1e-6):
    return 1 - _dice_loss_for_label(y_true, y_pred, 2, smooth)

def dice_coef_enhancing(y_true, y_pred, smooth=1e-6):
    return 1 - _dice_loss_for_label(y_true, y_pred, 3, smooth)

# Combined loss function that combines Binary cross entropy and Dice loss, weighted with alpha
def combined_loss_wrapper(class_weights, alpha=0.5, smooth=1e-5):
    def combined_loss(y_true, y_pred):
        nonlocal class_weights
        nonlocal alpha
        
        # Calculate binary cross entropy
        bce_fn = binary_cross_entropy_per_channel_wrapper(class_weights)
        bce = bce_fn(y_true, y_pred)

        # Calculate Dice loss
        dl_fn = dice_loss_wrapper(class_weights)
        dl = dl_fn(y_true, y_pred)

        # Calculate combined loss
        combined = alpha * bce + (1-alpha) * dl

        return combined
    return combined_loss