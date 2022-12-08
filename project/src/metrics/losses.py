"""This module contains loss functions and metrics."""

import numpy as np
from keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
import tensorflow as tf
from typing import List, Callable


def dice_coef(true: tf.Tensor, pred: tf.Tensor) -> float:
    """Calculate DICE coefficient."""
    return (2 * K.sum(true * pred) + 1) / (K.sum(true) + K.sum(pred) + 1)


def bce_dice_loss(true: tf.Tensor, pred: tf.Tensor) -> float:
    """Add Dice_loss to binary crossentropy."""
    return binary_crossentropy(true, pred) + 0.25 * (1 - dice_coef(true, pred))


def weighted_loss(original_loss_func: Callable, weights_list: List[float]) -> float:
    """Add weights classes in loss function."""
    def loss_func(true: tf.Tensor, pred: tf.Tensor) -> float:
        """Origin loss function."""
        axis = -1
        class_selectors = K.argmax(true, axis=axis)
        class_selectors = [
            K.equal(np.int64(i), class_selectors) for i in range(len(weights_list))
        ]
        class_selectors = [K.cast(x, K.floatx()) for x in class_selectors]
        weights = [sel * w for sel, w in zip(class_selectors, weights_list)]
        weight_multiplier = weights[0]
        for i in range(1, len(weights)):
            weight_multiplier = weight_multiplier + weights[i]
        loss = original_loss_func(true, pred)

        return loss * weight_multiplier

    return loss_func
