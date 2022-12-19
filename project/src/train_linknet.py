"""This module runs process of train model segmentation."""
import json
import pickle
import pandas as pd

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from project.src.metrics.losses import bce_dice_loss, dice_coef, weighted_loss
from project.src.models import get_linknet
from project.src.preprocessing.data_generator import DataGeneratorSeg
from project.src.preprocessing.split_data import train_val_split


# Получаем параметры для обучения
with open('project/experiments/params.json', 'r') as f:
    params = json.load(f)
params = params["segmentation"]

# Получаем данные для обучения
dataframe = pd.read_csv("project/data/train/train.csv")
with open('project/data/processed_data/train_seg_data.pickle', 'rb') as f:
    train = pickle.load(f)
with open('project/data/processed_data/test_seg_data.pickle', 'rb') as f:
    test = pickle.load(f)

# Создаем генераторы для обучающих и тестовых выборок
train_gen_seg = DataGeneratorSeg(train[:8], dataframe, batch_size = params["batch_size"])
test_gen_seg = DataGeneratorSeg(test[:8], dataframe, aug=False, batch_size = params["batch_size"])

lr_callback = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=10,
    verbose=0,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
)
save_callback = ModelCheckpoint(
    "project/weights/best_linknet.hdf5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="auto",
)
callbacks = [lr_callback, save_callback]

linknet = get_linknet()
class_weight = params["class_weight"]

linknet.compile(
    optimizer=Adam(params["learning_rate"]),
    loss=weighted_loss(bce_dice_loss, class_weight),
    metrics=[dice_coef],
)
history = linknet.fit(train_gen_seg, validation_data=test_gen_seg, epochs=params["epochs"], callbacks=callbacks)

with open('project/experiments/train_history_seg.json', 'w') as f:
    json.dump(str(history.history), f, indent=4)

print("Обучение закончено")
