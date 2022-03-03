import albumentations as A
import numpy as np


def augmentation(train=True):
    if train:
        transform = A.Compose([
            A.Resize(width=512, height=256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=(-10, 10), p=0.25,),
            ])
    else:
        transform = A.Compose([
            A.Resize(width=512, height=256),
            ])
    return transform


def get_mask(name, df):
    mask = np.zeros(1600*256)
    for class_id, pixels in df.loc[df.ImageId == name, ['ClassId', 'EncodedPixels']].values:
        if class_id == 0:  # Если дефектов нет, оставляем нули в маске
            break
        pixels = np.array(pixels.split(), dtype=np.int32)  # получаем данные из соответствующего столбца
        pixels[1::2] += pixels[::2]  # получаем индексы окончания строки маски
        for i in range(0, len(pixels), 2):
            mask[pixels[i]+1:pixels[i+1]+2] += class_id  # ставим номер класса в нужных пикселях
    mask = mask.reshape(1600, 256).T
    return mask
