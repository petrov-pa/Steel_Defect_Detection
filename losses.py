from tensorflow.keras import backend as K
from keras.losses import binary_crossentropy
import numpy as np


# Функция считает степень пересечения настоящей и предсказанной маски
def dice_coef(true, pred):
    """Считает коэффициент Дайса.
    Parameters:
    true -- Настоящая маска дефекта (numpy array)
    pred -- Предсказаная маска дефекта (numpy array)
      """
    return (2 * K.sum(true*pred)+1) / (K.sum(true) + K.sum(pred)+1)


# Функция потерь составная
def bce_dice_loss(true, pred):
    """Добавляет Dice_loss к бинарной кроссентропии.
    Parameters:
    true -- Настоящая маска дефекта (numpy array)
    pred -- Предсказаная маска дефекта (numpy array)
      """
    return binary_crossentropy(true, pred) + 0.25*(1-dice_coef(true, pred))


def weighted_loss(original_loss_func, weights_list):

    def loss_func(true, pred):

        axis = -1  # if channels last
        # axis=  1 # if channels first

        # argmax returns the index of the element with the greatest value
        # done in the class axis, it returns the class index
        class_selectors = K.argmax(true, axis=axis)
        # if your loss is sparse, use only true as classSelectors

        # considering weights are ordered by class, for each class
        # true(1) if the class index is equal to the weight index
        class_selectors = [K.equal(np.int64(i), class_selectors) for i in range(len(weights_list))]

        # casting boolean to float for calculations
        # each tensor in the list contains 1 where ground true class is equal to its index
        # if you sum all these, you will get a tensor full of ones.
        class_selectors = [K.cast(x, K.floatx()) for x in class_selectors]

        # for each of the selections above, multiply their respective weight
        weights = [sel * w for sel, w in zip(class_selectors, weights_list)]

        # sums all the selections
        # result is a tensor with the respective weight for each element in predictions
        weight_multiplier = weights[0]
        for i in range(1, len(weights)):
            weight_multiplier = weight_multiplier + weights[i]
        # make sure your originalLossFunc only collapses the class axis
        # you need the other axes intact to multiply the weights tensor
        loss = original_loss_func(true, pred)
        loss = loss * weight_multiplier

        return loss
    return loss_func
