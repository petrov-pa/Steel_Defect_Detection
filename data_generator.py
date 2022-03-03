from tensorflow.keras.utils import to_categorical, Sequence
import numpy as np
import cv2
from preprocessing import get_mask, augmentation


class DataGeneratorSeg(Sequence):
    def __init__(self, img_name, data, batch_size=8, shuffle=True, aug=True):
        self.img_name = img_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug
        self.data = data
        self.on_epoch_end()
        self.indexes = np.arange(len(self.img_name))

    def __len__(self):
        # Получаем количество эпох
        return int(np.floor(len(self.img_name) / self.batch_size))

    def __getitem__(self, index):
        # Получаем индекса батча данных
        list_IDs_temp = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Получаем данные по индексу
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        # Обновляем и перемешиваем индексы в конце каждой эпохи
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        batch_imgs = list()
        batch_labels = list()

        for i in list_IDs_temp:
            # Загружаем данные и меняем размер изображений
            path_img = '/content/train_images/ ' + self.img_name[i]
            img = cv2.imread(path_img)
            label = get_mask(self.img_name[i], self.data)

            # Выполняем аугментацию для обучающей выборки
            if self.aug:
                transform = augmentation(train=True)
                transformed = transform(image=img, mask=label)
                img = transformed['image']
                label = transformed['mask']
            else:
                transform_test = augmentation(train=False)
                transformed = transform_test(image=img, mask=label)
                img = transformed['image']
                label = transformed['mask']
                # Делаем One-Hot-Encoding маски
            labels = np.zeros((256, 512, 4))
            for j in range(4):
                labels[:, :, j] = (label == j + 1).astype(np.int32, copy=False)
            # Добавляем в список
            batch_labels.append(labels)
            batch_imgs.append(img)

        return np.array(batch_imgs), np.array(batch_labels)


class DataGeneratorClf(Sequence):
    def __init__(self, img_name, dict_label, batch_size=8, shuffle=True, aug=True):
        self.img_name = img_name
        self.dict_label = dict_label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug
        self.on_epoch_end()
        self.indexes = np.arange(len(self.img_name))

    def __len__(self):
        # Получаем количество эпох
        return int(np.floor(len(self.img_name) / self.batch_size))

    def __getitem__(self, index):
        # Получаем индекса батча данных
        list_IDs_temp = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Получаем данные по индексу
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        # Обновляем и перемешиваем индексы в конце каждой эпохи
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        batch_imgs = list()
        batch_labels = list()
        for i in list_IDs_temp:
            # Загружаем данные и меняем размер изображения
            path_img = '/content/train_images/' + self.img_name[i]
            img = cv2.imread(path_img)
            # Получаем метку классификатора
            labels = self.dict_label[self.img_name[i]]
            # Выполняем аугментацию для обучающей выборки
            if self.aug:
                transform = augmentation(train=True)
                transformed = transform(image=img, mask=np.zeros((256, 512, 3)), )
                img = transformed['image']
            else:
                transform_test = augmentation(train=False)
                transformed = transform_test(image=img, mask=np.zeros((256, 512, 3)))
                img = transformed['image']
            batch_labels.append(labels)
            batch_imgs.append(img)

        return np.array(batch_imgs), np.array(batch_labels)
