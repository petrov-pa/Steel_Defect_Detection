import numpy as np
from models import get_clf, get_linknet
import cv2
import os
from albumentations import Normalize


def run():
    list_img = os.listdir('./inputs')
    if not list_img:
        return print('Нет файлов в папке')
    # загружаем обученную модель
    clf = get_clf()
    linknet = get_linknet()
    clf.load_weights('./models/best_classifier.hdf5')
    linknet.load_weights('./models/best_linknet.hdf5')
    normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    for img in list_img:
        orig_img = cv2.imread('./inputs/' + img)
        if orig_img is None:
            return print('Неверный формат файла: {}'.format(img))
        img_for_clf = normalize(image=orig_img)['image']
        img_for_seg = orig_img/255
        # предсказываем наличие дефекта
        pred_clf = clf.predict(np.array([img_for_clf[:, 320 * part:320 * (part + 1), :] for part in range(5)]))
        if np.sum(pred_clf > 0.5)+1:  # если дефект есть, то делаем сегментацию
            pred_seg = linknet.predict(np.array([img_for_seg[:, 320 * part:320 * (part + 1), :] for part in range(5)]))
            full_pred = np.hstack([pred_seg[0], pred_seg[1], pred_seg[2], pred_seg[3], pred_seg[4]])
            pred_mask = (full_pred > 0.5).astype(np.int32)
            pred_mask = pred_mask * 200
            pred_mask[:, :, 0] = pred_mask[:, :, 0] + pred_mask[:, :, 3]
            pred_mask = pred_mask[:, :, :-1]
        else:
            pred_mask = np.zeros((256, 1600, 1))
        # записываем в файл
        cv2.imwrite('./outputs/' + img, orig_img+pred_mask)
    return print('Сегментация закончена')


run()
