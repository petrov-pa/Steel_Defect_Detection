"""This module runs process of train classifier."""

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from project.src.models import get_clf
from project.src.preprocessing.preprocessing import DataGenerator
from project.src.preprocessing.split_data import train_val_split

# Получаем данные для обучения
_, _, train, test, dataframe = train_val_split()

# Создаем генераторы для обучающих и тестовых выборок
train_gen_clf = DataGenerator(train, dataframe, seg=False)
test_gen_clf = DataGenerator(test, dataframe, seg=False, aug=False)

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
    "./weights/best_classifier.hdf5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="auto",
)
callbacks = [lr_callback, save_callback]

clf = get_clf()
clf.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
clf.fit(train_gen_clf, validation_data=test_gen_clf, epochs=80, callbacks=callbacks)
