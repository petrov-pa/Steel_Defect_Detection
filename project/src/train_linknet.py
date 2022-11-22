from tensorflow.keras.optimizers import Adam
from project.src.preprocessing.split_data import train_val_split
from project.src.preprocessing.preprocessing import DataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from project.src.models import get_linknet
from project.src.metrics.losses import dice_coef, bce_dice_loss, weighted_loss

# Получаем данные для обучения
train, test, _, _, df = train_val_split()

# Создаем генераторы для обучающих и тестовых выборок
train_gen_seg = DataGenerator(train, df)
test_gen_seg = DataGenerator(test, df, aug=False)

lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=0,
                                mode="auto", min_delta=0.0001, cooldown=0, min_lr=0)
save_callback = ModelCheckpoint('./weights/best_linknet_v1.hdf5', monitor='val_loss', verbose=1,
                                save_best_only=True, mode='auto')
callbacks = [lr_callback, save_callback]

linknet = get_linknet()
class_weight = [1., 10., 0.05, 0.25]

linknet.compile(optimizer=Adam(0.001), loss=weighted_loss(bce_dice_loss, class_weight), metrics=[dice_coef])
linknet.fit(train_gen_seg, validation_data=test_gen_seg, epochs=60, callbacks=callbacks)

