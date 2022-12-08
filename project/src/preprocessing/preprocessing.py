"""This module contains preprocessing functions."""

import numpy as np
import pandas as pd


def get_mask(name: str, dataframe: pd.DataFrame) -> np.ndarray:
    """Convert the string representation of the defect to a mask."""
    mask = np.zeros(1600 * 256)
    for class_id, pixels in dataframe.loc[
        dataframe.ImageId == name, ["ClassId", "EncodedPixels"]
    ].to_array():
        if class_id == 0:  # Если дефектов нет, оставляем нули в маске
            break
        pixels = np.array(pixels.split(), dtype=np.int32)
        pixels[1::2] += pixels[
            ::2
        ]  # меняем значение длины последовательности на номер последнего пикселя
        for i in range(0, len(pixels), 2):
            mask[
                pixels[i] + 1: pixels[i + 1] + 2
            ] += class_id  # ставим номер класса в нужных пикселях
    mask = mask.reshape(1600, 256).T
    return mask
