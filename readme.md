# Сегментация дефектов металлопроката

## Цель проекта: 
Обучить нейронную сеть предсказывать местоположение и тип дефектов, найденных в производстве стали.

## Датасет:
Набор содержит 12600 изображений формата 1600х256 пикселей. Каждое изображение может не иметь дефектов, иметь дефект одного класса или дефекты нескольких классов.

## Пареметры сети:
Архитектура - U-net c дополнительным выходом классификатора. В качестве backbone сеть EfficientNetB1

Оптимизатор - Adam

Функция ошибки - binary_crossentropy + dice_loss

Метрика - dice_coef для сегментации, accuracy для классификации

## Обученные модели и результаты:
Val_loss: 0.1180  
Val_dice_coef: 0.6675

![загруженное](https://user-images.githubusercontent.com/64748758/145363218-2924c276-9690-4377-97d9-6e7a380dbb86.png)

![загруженное (1)](https://user-images.githubusercontent.com/64748758/145363234-6b5e05f9-b589-4367-ad80-31ae8bc5b8a5.png)