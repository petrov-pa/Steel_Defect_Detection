from tensorflow.keras import backend as K
from keras.losses import binary_crossentropy


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
