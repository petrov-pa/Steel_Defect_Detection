from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
from src.preprocessing.preprocessing import get_mask, augmentation


class DataGenerator(Sequence):
    def __init__(self, img_name, img_data, batch_size=8, shuffle=True, aug=True, seg=True):
        self.img_name = img_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug
        self.seg = seg
        self.img_data = img_data
        self.transform = augmentation()
        self.on_epoch_end()

    def __len__(self):
        # Получаем количество эпох
        return int(np.floor(len(self.img_name) / self.batch_size))

    def __getitem__(self, index):
        # Получаем индекса батча данных
        list_id_temp = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Получаем данные по индексу
        X, y = self.__data_generation(list_id_temp)
        return X, y

    def on_epoch_end(self):
        # Обновляем и перемешиваем индексы в конце каждой эпохи
        self.indexes = np.arange(len(self.img_name))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def augmentations(self, image, label):
        # Выполняем аугментацию
        if self.aug:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        return image, label

    def __data_generation(self, list_id_temp):
        batch_imgs = list()
        batch_labels = list()

        for i in list_id_temp:
            # Загружаем данные и меняем размер изображений
            path_img = '/content/train_images/' + self.img_name[i]
            img = cv2.imread(path_img) / 255
            label = get_mask(self.img_name[i], self.img_data)

            for part in range(5):
                img_part = img[:, 320 * part:320 * (part + 1), :]
                label_part = label[:, 320 * part:320 * (part + 1)]
                img_aug, label_aug = self.augmentations(img_part, label_part)
                if self.seg:  # Если обучаем сегментации, делаем One-Hot-Encoding маски и добавляем в список
                    if label_aug.sum():
                        label_ohe = np.zeros((256, 320, 4))
                        for class_id in range(4):
                            label_ohe[:, :, class_id] = (label_aug == class_id + 1).astype(np.int32, copy=False)
                        batch_labels.append(label_ohe)
                        batch_imgs.append(img_aug)
                else:
                    label_clf = 1 if label_aug.sum() else 0
                    batch_labels.append(label_clf)
                    batch_imgs.append(img_aug)

        return np.array(batch_imgs), np.array(batch_labels)
