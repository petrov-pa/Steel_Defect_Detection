from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Activation
from segmentation_models import Linknet
from tensorflow.keras.models import Model
# import keras.layers as layers
from tensorflow.keras import backend as K


def get_clf():
    """Подгружает модель, добавляет слои на выход и
    возвращает модель классификации """
    model = Linknet(backbone_name='efficientnetb2', classes=4, encoder_weights=None, encoder_freeze=False)
    x = model.layers[331].output
    x = GlobalAveragePooling2D(keepdims=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    clf = Model(model.input, output)
    return clf


def get_linknet():
    """Подгружает модель сегментации """
    linknet = Linknet(backbone_name='efficientnetb2', classes=4, encoder_weights=None, encoder_freeze=False)
    return linknet


class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)
