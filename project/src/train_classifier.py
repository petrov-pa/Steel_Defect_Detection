"""This module runs process of train classifier."""
import json

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from project.src.models import get_clf
from project.src.preprocessing.data_generator import DataGeneratorClf
from project.src.preprocessing.split_data import train_val_split

# Получаем параметры для обучения
with open('../experiments/params.json', 'r') as f:
    params = json.load(f)
params = params["classifier"]

# Получаем данные для обучения
_, _, train, test, dataframe = train_val_split()

# Создаем генераторы для обучающих и тестовых выборок
train_gen_clf = DataGeneratorClf(train, dataframe, batch_size = params["batch_size"])
test_gen_clf = DataGeneratorClf(test, dataframe, aug=False, batch_size = params["batch_size"])

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
    "../weights/best_classifier_1.hdf5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="auto",
)
callbacks = [lr_callback, save_callback]

clf = get_clf()
clf.compile(optimizer=Adam(params["learning_rate"]), loss=params["loss"], metrics=params["metrics"])
history = clf.fit(train_gen_clf, validation_data=test_gen_clf, epochs=params["epochs"], callbacks=callbacks)

with open('../experiments/train_history_clf.json', 'w') as f:
    json.dump(history.history, f, indent=4)
