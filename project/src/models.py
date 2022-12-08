"""This module defines the architecture of neural networks."""

import tensorflow as tf
from segmentation_models import Linknet
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model


def get_clf() -> Model:
    """Load segmentation model and return classifier model."""
    model = Linknet(
        backbone_name="efficientnetb2",
        classes=4,
        encoder_weights=None,
        encoder_freeze=False,
    )
    x = model.layers[331].output
    x = GlobalAveragePooling2D(keepdims=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)
    return Model(model.input, output)


def get_linknet() -> Model:
    """Load segmentation model."""
    return Linknet(
        backbone_name="efficientnetb2",
        classes=4,
        encoder_weights=None,
        encoder_freeze=False,
    )


class FixedDropout(Dropout):
    """
    Wrapper over custom dropout. Fix problem of ``None`` shape for tf.keras.

    It is not possible to define FixedDropout class as global object,
    because we do not have modules for inheritance at first time.
    """

    def _get_noise_shape(self, inputs: tf.Tensor) -> tuple:
        """Change shape tensor."""
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)
