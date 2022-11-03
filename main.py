import numpy as np
from src.models import FixedDropout
import cv2
import os
from tensorflow.keras.models import load_model


def main():
    list_img = os.listdir('./data/test')
    if not list_img:
        return print('Нет файлов в папке')
    # загружаем обученную модель
    clf = load_model('weights/clf.h5', custom_objects={'FixedDropout': FixedDropout}, compile=False)
    linknet = load_model('weights/linknet.h5', custom_objects={'FixedDropout': FixedDropout}, compile=False)
    for name in list_img:
        orig_img = cv2.imread('./data/test/' + name)
        if orig_img is None:
            return print('Неверный формат файла: {}'.format(name))
        norm_img = orig_img/255
        img_part = [norm_img[:, 320 * part:320 * (part + 1), :] for part in range(5)]
        # предсказываем наличие дефекта
        pred_clf = clf.predict(np.array(img_part))
        if np.sum(pred_clf > 0.5):  # если дефект есть, то делаем сегментацию
            pred_seg = linknet.predict(np.array(img_part))
            full_pred = np.hstack([pred_seg[0], pred_seg[1], pred_seg[2], pred_seg[3], pred_seg[4]])
            pred_mask = (full_pred > 0.5).astype(np.int32)
            pred_mask = pred_mask * 100
            pred_mask[:, :, 0] = pred_mask[:, :, 0] + pred_mask[:, :, 3]
            pred_mask = pred_mask[:, :, :-1]
        else:
            pred_mask = np.zeros((256, 1600, 1))
        # записываем в файл
        cv2.imwrite('./outputs/' + name, orig_img+pred_mask)
    return print('Сегментация закончена')


if __name__ == '__main__':
    main()
