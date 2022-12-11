"""This module contains function for split data."""

import os

import numpy as np
import pandas as pd


def train_val_split() -> np.ndarray:
    """Split sample into train and valid subsets."""
    train_data = "../data/train/train_images/"
    dataframe = pd.read_csv("../data/train/train.csv")
    # Добавим изображения без дефектов в таблицу
    for name in os.listdir(train_data):
        if name not in dataframe.ImageId.to_list():
            new_row = {"ImageId": name, "ClassId": 0, "EncodedPixels": "1 409600"}
            dataframe = dataframe.append(new_row, ignore_index=True)

    # Добавим в таблицу столбец с меткой есть ли дефект или нет
    dataframe["defect"] = np.zeros(dataframe.shape[0])
    dataframe.loc[dataframe.ClassId != 0, "defect"] = 1
    count_to_img = dataframe.ImageId.value_counts()
    test_seg = []  # тестовая выборка для задачи сегментации
    test_seg.extend(
        count_to_img[count_to_img == 2][:85].index.to_list()
    )  # добавим изображения с двумя дефектами

    # отберем изображения, которых еще нет в тестовой выборке
    df_to_split = dataframe.loc[~dataframe.ImageId.isin(test_seg)]
    # добавим недостающее количество изображений в выборку
    test_seg.extend(df_to_split.loc[df_to_split.ClassId == 1, "ImageId"][:153].to_list())
    test_seg.extend(df_to_split.loc[df_to_split.ClassId == 2, "ImageId"][:36].to_list())
    test_seg.extend(df_to_split.loc[df_to_split.ClassId == 3, "ImageId"][:951].to_list())
    test_seg.extend(df_to_split.loc[df_to_split.ClassId == 4, "ImageId"][:102].to_list())
    df_to_seg = dataframe.loc[
        ~dataframe.ImageId.isin(test_seg)
    ]  # убираем из набора данных тестовые изображения
    df_to_seg = df_to_seg.drop(
        df_to_seg.loc[df_to_seg.ClassId == 0].index[:], axis=0
    )  # убираем изображения без дефектов
    # отбираем названия изображений для обучающей выборки модели сегментации
    train_seg = df_to_seg["ImageId"].to_list()

    # создадим тестовую выборку для классификатора
    test_clf = test_seg.copy()
    test_clf.extend(df_to_split.loc[df_to_split.ClassId == 0, "ImageId"][:1180].to_list())
    # отбираем названия изображений для обучающей выборки классификатора
    train_clf = dataframe.loc[~dataframe.ImageId.isin(test_clf), "ImageId"]

    return train_seg, test_seg, train_clf.to_list(), test_clf, dataframe
