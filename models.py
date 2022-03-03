from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from segmentation_models import Unet
from tensorflow.keras.models import Model


def get_model():
    unet = Unet(backbone_name='efficientnetb1', classes=4, encoder_weights=None, encoder_freeze=False)
    x = unet.layers[-42].output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    clf = Model(unet.input, output)
    model = Model(unet.input, [unet.output, output])
    return clf, unet, model
