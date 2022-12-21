"""This module runs process of evaluate classifier."""
import json
import pickle
import pandas as pd

from tensorflow.keras.optimizers import Adam

from project.src.models import get_clf
from project.src.preprocessing.data_generator import DataGeneratorClf


# Получаем параметры для обучения
with open('project/experiments/params.json', 'r') as f:
    params = json.load(f)
params = params["classifier"]

# Получаем данные для обучения
dataframe = pd.read_csv("project/data/train/train.csv")
with open('project/data/processed_data/test_clf_data.pickle', 'rb') as f:
    test = pickle.load(f)

# Создаем генераторы для обучающих и тестовых выборок
test_gen_clf = DataGeneratorClf(test[:16], dataframe, aug=False, batch_size = params["batch_size"])

clf = get_clf()
clf.compile(optimizer=Adam(params["learning_rate"]), loss=params["loss"], metrics=params["metrics"])
clf.load_weights("project/weights/best_classifier.hdf5")
score = clf.evaluate(test_gen_clf)
score = {params["loss"]: score[0], params["metrics"][0]: score[1]}
with open('project/experiments/clf_metrics.json', 'w') as f:
    json.dump(score, f, indent=4)
