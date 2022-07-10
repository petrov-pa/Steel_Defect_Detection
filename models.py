from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Activation
from segmentation_models import Linknet
from tensorflow.keras.models import Model


def get_clf():
    """Подгружает модель, добавляет слои на выход и
    возвращает модель классификации """
    model = Linknet(backbone_name='efficientnetb2', classes=4, encoder_weights=None, encoder_freeze=False)
    x = model.layers[331].output
    x = GlobalAveragePooling2D()(x)
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
