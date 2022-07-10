import pandas as pd
import os
import numpy as np


def train_val_split():
    """ Разбивает выборку на обучающую и валидационную
    Returns:
        train_seg - Список с названиями файлов обучающей выборки модели сегментации (np.array)
        test_seg - Список с названиями файлов валидационной выборки модели сегментации (np.array)
        train_clf - Список с названиями файлов обучающей выборки модели классификации (np.array)
        test_clf - Список с названиями файлов валидационной выборки модели классификации (np.array)
        df - Таблица с информацией о разметке (DataFrame)

     """
    train_data = './data/train/images/'
    df = pd.read_csv('./data/train/train.csv')
    # В выборке есть много изображений, которые не содержат дефектов. Добавим их в нашу таблицу
    for name in os.listdir(train_data):
        if name not in df.ImageId.values:
            new_row = {'ImageId': name, 'ClassId': 0, 'EncodedPixels': '1 409600'}
            df = df.append(new_row, ignore_index=True)

    # Добавим в таблицу столбец с меткой есть ли дефект или нет
    df['defect'] = np.zeros(df.shape[0])
    df.loc[df.ClassId != 0, 'defect'] = 1
    count_to_img = df.ImageId.value_counts()
    test_seg = []  # тестовая выборка для задачи сегментации
    test_seg.extend(count_to_img[count_to_img == 2][:85].index.values)  # добавим изображения с двумя дефектами

    # отберем изображения, которых еще нет в тестовой выборке
    df_to_split = df.loc[~df.ImageId.isin(test_seg)]
    # добавим недостающее количество изображений в выборку
    test_seg.extend(df_to_split.loc[df_to_split.ClassId == 1, 'ImageId'][:153].values)
    test_seg.extend(df_to_split.loc[df_to_split.ClassId == 2, 'ImageId'][:36].values)
    test_seg.extend(df_to_split.loc[df_to_split.ClassId == 3, 'ImageId'][:951].values)
    test_seg.extend(df_to_split.loc[df_to_split.ClassId == 4, 'ImageId'][:102].values)
    df_to_seg = df.loc[~df.ImageId.isin(test_seg)]  # убираем из набора данных тестовые изображения
    df_to_seg.drop(df_to_seg.loc[df_to_seg.ClassId == 0].index[:], axis=0,
                   inplace=True)  # убираем изображения без дефектов
    # отбираем названия изображений для обучающей выборки модели сегментации
    train_seg = df_to_seg['ImageId'].values

    # создадим тестовую выборку для классификатора
    test_clf = test_seg.copy()
    test_clf.extend(df_to_split.loc[df_to_split.ClassId == 0, 'ImageId'][:1180].values)
    # отбираем названия изображений для обучающей выборки классификатора
    train_clf = df.loc[~df.ImageId.isin(test_clf), 'ImageId'].values

    return train_seg, test_seg, train_clf, test_clf, df
