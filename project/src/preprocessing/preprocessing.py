import albumentations as A
import numpy as np


def augmentation():
    """ Return transformations pipeline. """
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(-10, 10), p=0.25,),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    return transform


def get_mask(name, df):
    """Converts the string representation of the defect to a mask.
    Parameters:
    name -- file name (str)
    df -- dataframe with mask information (DataFrame)
    Returns:
        numpy array
    """
    mask = np.zeros(1600*256)
    for class_id, pixels in df.loc[df.ImageId == name, ['ClassId', 'EncodedPixels']].values:
        if class_id == 0:  # Если дефектов нет, оставляем нули в маске
            break
        pixels = np.array(pixels.split(), dtype=np.int32)
        pixels[1::2] += pixels[::2]  # меняем значение длины последовательности на номер последнего пикселя
        for i in range(0, len(pixels), 2):
            mask[pixels[i]+1:pixels[i+1]+2] += class_id  # ставим номер класса в нужных пикселях
    mask = mask.reshape(1600, 256).T
    return mask
