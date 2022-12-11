"""This module contains data generator for augmentations and train on batch."""

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from typing import List
import albumentations as A

from project.src.preprocessing.preprocessing import get_mask


class DataGeneratorClf(Sequence):
    """Return data generator for train models classifications."""

    def __init__(
        self, img_name: str, img_data: str, batch_size: int = 8,
        shuffle: bool = True, aug: bool = True
    ):
        self.img_name = img_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug
        self.img_data = img_data
        self.on_epoch_end()

    def __len__(self):
        """Return number of epochs."""
        return int(np.floor(len(self.img_name) / self.batch_size))

    def __getitem__(self, index: int) -> (np.ndarray, np.ndarray):
        """Return data on index."""
        # Получаем индексы батча данных
        list_id_temp = self.indexes[
            index * self.batch_size: (index + 1) * self.batch_size
        ]
        # Получаем данные по индексу
        x, y = self.__data_generation(list_id_temp)
        return x, y

    def on_epoch_end(self) -> None:
        """Shuffle index on epoch end."""
        self.indexes = np.arange(len(self.img_name))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def augmentations(self, image: np.ndarray, label: np.ndarray) -> (np.ndarray, np.ndarray):
        """Transform image."""
        if self.aug:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=(-10, 10), p=0.25)
                ])
            transformed = transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]
        return image, label

    def __data_generation(self, list_id_temp: List[int]) -> (np.ndarray, np.ndarray):
        """Return batch images and labels."""
        batch_imgs = []
        batch_labels = []

        for i in list_id_temp:
            # Загружаем данные и меняем размер изображений
            path_img = "../data/train/train_images/" + self.img_name[i]
            img = cv2.imread(path_img) / 255
            label = get_mask(self.img_name[i], self.img_data)

            for part in range(5):
                img_part = img[:, 320 * part: 320 * (part + 1), :]
                label_part = label[:, 320 * part: 320 * (part + 1)]
                img_aug, label_aug = self.augmentations(img_part, label_part)
                label_clf = 1 if label_aug.sum() else 0
                batch_labels.append(label_clf)
                batch_imgs.append(img_aug)

        return np.array(batch_imgs), np.array(batch_labels)


class DataGeneratorSeg(DataGeneratorClf):
    """Return data generator for train segmentations model."""

    def __init__(
        self, img_name: str, img_data: str, batch_size: int = 8,
        shuffle: bool = True, aug: bool = True
    ):
        self.img_name = img_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug
        self.img_data = img_data
        self.on_epoch_end()

    def __data_generation(self, list_id_temp: List[int]) -> (np.ndarray, np.ndarray):
        """Return batch images and labels."""
        batch_imgs = []
        batch_labels = []

        for i in list_id_temp:
            # Загружаем данные и меняем размер изображений
            path_img = "../data/train/train_images/" + self.img_name[i]
            img = cv2.imread(path_img) / 255
            label = get_mask(self.img_name[i], self.img_data)

            for part in range(5):
                img_part = img[:, 320 * part: 320 * (part + 1), :]
                label_part = label[:, 320 * part: 320 * (part + 1)]
                img_aug, label_aug = self.augmentations(img_part, label_part)
                if label_aug.sum():
                    label_ohe = np.zeros((256, 320, 4))
                    for class_id in range(4):
                        label_ohe[:, :, class_id] = (
                            label_aug == class_id + 1
                        ).astype(np.int32, copy=False)
                    batch_labels.append(label_ohe)
                    batch_imgs.append(img_aug)

        return np.array(batch_imgs), np.array(batch_labels)
