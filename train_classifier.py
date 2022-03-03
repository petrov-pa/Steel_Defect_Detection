from tensorflow.keras.optimizers import Adam
from split_data import train_test_split
from preprocessing import augmentation
from data_generator import DataGeneratorClf
from models import get_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# Получаем данные для обучения
_, _, train, test, df = train_test_split()
# Создаем словарь {Название изображения: наличие дефекта}
dict_label = dict()
for value in df.values:
    dict_label[value[0]] = value[3]

# Создаем трансформер для данных
transform = augmentation(train=True)
transform_test = augmentation(train=False)
# Создаем генераторы для обучающих и тестовых выборок
train_gen_clf = DataGeneratorClf(train, dict_label)
test_gen_clf = DataGeneratorClf(test, dict_label, aug=False)

lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=0,
                                mode="auto", min_delta=0.0001, cooldown=0, min_lr=0)
save_callback = ModelCheckpoint('./models/best_classifier.hdf5', monitor='val_loss', verbose=1,
                                save_best_only=True, mode='auto')
callbacks = [lr_callback, save_callback]

clf, _, _ = get_model()
clf.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
clf.fit(train_gen_clf, validation_data=test_gen_clf, epochs=80, callbacks=callbacks)
