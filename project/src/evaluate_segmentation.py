"""This module runs process of evaluate model segmentation."""
import json
import pickle
import pandas as pd

from tensorflow.keras.optimizers import Adam

from project.src.metrics.losses import bce_dice_loss, dice_coef, weighted_loss
from project.src.models import get_linknet
from project.src.preprocessing.data_generator import DataGeneratorSeg


# Получаем параметры для обучения
with open('project/experiments/params.json', 'r') as f:
    params = json.load(f)
params = params["segmentation"]

# Получаем данные для обучения
dataframe = pd.read_csv("project/data/train/train.csv")
with open('project/data/processed_data/test_seg_data.pickle', 'rb') as f:
    test = pickle.load(f)

# Создаем генераторы для обучающих и тестовых выборок
test_gen_seg = DataGeneratorSeg(test[:8], dataframe, aug=False, batch_size = params["batch_size"])

linknet = get_linknet()
class_weight = params["class_weight"]

linknet.compile(
    optimizer=Adam(params["learning_rate"]),
    loss=weighted_loss(bce_dice_loss, class_weight),
    metrics=[dice_coef],
)
linknet.load_weights("project/weights/best_linknet.hdf5")
score = linknet.evaluate(train_gen_seg)

score = {"bce_dice_loss": score[0], "dice_coef": score[1]}
with open('project/experiments/seg_metrics.json', 'w') as f:
    json.dump(score, f, indent=4)


with open('project/experiments/train_history_seg.json', 'w') as f:
    json.dump(str(history.history), f, indent=4)

print("Обучение закончено")