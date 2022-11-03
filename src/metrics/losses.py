from tensorflow.keras import backend as K
from keras.losses import binary_crossentropy
import numpy as np


# Функция считает степень пересечения настоящей и предсказанной маски
def dice_coef(true, pred):
    """Calculate DICE coefficient.
    Parameters:
    true -- True mask (numpy array)
    pred -- Predict mask (numpy array)
      """
    return (2 * K.sum(true*pred)+1) / (K.sum(true) + K.sum(pred)+1)


# Функция потерь составная
def bce_dice_loss(true, pred):
    """Adds Dice_loss to binary crossentropy.
    Parameters:
    true -- True mask (numpy array)
    pred -- Predict mask (numpy array)
      """
    return binary_crossentropy(true, pred) + 0.25*(1-dice_coef(true, pred))


def weighted_loss(original_loss_func, weights_list):

    def loss_func(true, pred):

        axis = -1
        class_selectors = K.argmax(true, axis=axis)
        class_selectors = [K.equal(np.int64(i), class_selectors) for i in range(len(weights_list))]
        class_selectors = [K.cast(x, K.floatx()) for x in class_selectors]
        weights = [sel * w for sel, w in zip(class_selectors, weights_list)]
        weight_multiplier = weights[0]
        for i in range(1, len(weights)):
            weight_multiplier = weight_multiplier + weights[i]
        loss = original_loss_func(true, pred)
        loss = loss * weight_multiplier

        return loss
    return loss_func
