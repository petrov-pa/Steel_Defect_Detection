from tensorflow.keras.optimizers import Adam
from split_data import train_test_split
from preprocessing import augmentation
from data_generator import DataGeneratorSeg
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K
from keras.losses import binary_crossentropy
from models import get_model

# Получаем данные для обучения
train, test, _, _, df = train_test_split()
# Создаем трансформер для данных
transform = augmentation(train=True)
transform_test = augmentation(train=False)
# Создаем генераторы для обучающих и тестовых выборок
train_gen_seg = DataGeneratorSeg(train, df)
test_gen_seg = DataGeneratorSeg(test, df, aug=False)

lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=0,
                                mode="auto", min_delta=0.0001, cooldown=0, min_lr=0)
save_callback = ModelCheckpoint('./models/best_unet.hdf5', monitor='val_loss', verbose=1,
                                save_best_only=True, mode='auto')
callbacks = [lr_callback, save_callback]


# Функция считает степень пересечения настоящей и предсказанной маски
def dice_coef(true, pred):
    return (2 * K.sum(true*pred)+1) / (K.sum(true) + K.sum(pred)+1)


# Функция потерь составная
def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + 0.25*(1-dice_coef(y_true, y_pred))


clf, unet, _ = get_model()
clf.load_weights('./models/best_clf.hdf5')
for layer in clf.layers:
    layer.trainable = False
unet.compile(optimizer=Adam(0.001), loss=bce_dice_loss, metrics=[dice_coef])
unet.fit(train_gen_seg, validation_data=test_gen_seg, epochs=60, callbacks=callbacks)
