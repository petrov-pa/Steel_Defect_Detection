import numpy as np
from models import get_model
import cv2
import os


def run():
    list_img = os.listdir('./data/prediction/images')
    if not list_img:
        return 'Нет файлов в папке'
    images = []
    for img in list_img:
        img = cv2.imread('./data/prediction/images/' + img) / 255
        img = cv2.resize(img, (512, 256), cv2.INTER_NEAREST)
        images.append(img)
    # загружаем обученную модель
    clf, unet, model = get_model()
    unet.load_weights('./models/best_unet.hdf5')
    clf.load_weights('./models/best_clf.hdf5')
    # предсказываем дефекты
    predict = model.predict(np.array(images))
    unet_pred = predict[0]
    clf_pred = predict[1]
    # записываем в файл
    for ind in range(len(list_img)):
        if np.array(clf_pred[ind]) > 0.5:
            pred_mask = (unet_pred[ind] > 0.5).astype(np.int32)
            pred_mask = pred_mask[:, :, 0] * 50 + pred_mask[:, :, 1] * 100 + pred_mask[:, :, 2] * 150 + pred_mask[:, :, 3] * 200
            pred_mask = pred_mask.astype(np.float32)
            pred_mask = cv2.resize(pred_mask, (1600, 256), cv2.INTER_NEAREST)
            pred_mask[(pred_mask != 50) & (pred_mask != 100) & (pred_mask != 150) & (pred_mask != 200)] = 0
        else:
            pred_mask = np.zeros((256, 1600))
        cv2.imwrite('./data/prediction/mask/' + list_img[ind] + '.jpg', pred_mask)
    return print('Сегментация закончена')


run()
