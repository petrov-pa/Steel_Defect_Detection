# Сегментация дефектов металлопроката

## Постановка задачи: 
Сталь - один из важнейших строительных материалов современности. Стальные здания устойчивы 
к естественному и искусственному износу, что сделало этот материал повсеместным во всем мире. 

Процесс производства плоского стального листа достаточно сложный. До отправки стальной лист проходит путь от нагрева и прокатки до сушки и резки. На каждом этапе происходит взаимодействие со станками и возможно возникновение дефектов.

Целью проекта является предсказание расположения и типа дефектов. Такие предсказания помогут работникам отбраковывать изделия на ранних этапах, фиксировать дефекты, повторяющиеся на одних и тех же местах, а также настраивать оборудование, чтобы они не повторялись

## Датасет:
Обучающая выборка содержит 12600 изображений формата 1600х256 пикселей. Изображение может не иметь дефектов, иметь дефект одного класса или дефекты нескольких классов.
Тестовая выборка содержит 5500 изображений.

## Анализ данных:
   * Распределение изображений в зависимости от количества дефектов:
     <br> один тип дефекта - 6239 изображений 
     <br> два типа дефектов - 425 изображений 
     <br> три тип дефектов - 2 изображения 
   * Распределение изображений по классу дефекта:
     <br> 1 класс дефекта - 897 изображений 
     <br> 2 класс дефекта - 247 изображений 
     <br> 3 класс дефекта - 5150 изображения 
     <br> 4 класс дефекта - 801 изображения
   * Наличие дефекта на изображении:
     <br> Есть дефект - 6666 изображений 
     <br> Нет дефекта - 5902 изображений 
   * Распределение пикселей в зависимости от класса дефекта:
     <br> 1 класс - 3912129 пикселей 
     <br> 2 класс - 834471 пикселей     <br> 3 класс - 131306899 пикселей 
     <br> 4 класс - 27533572 пикселей 
     <br> Нет дефекта - 2742524929 пикселей
   * Пример разметки дефекта:
![image](https://user-images.githubusercontent.com/64748758/199672754-8d87a60d-66c6-4736-92d4-30e2fd830c20.png)
![image](https://user-images.githubusercontent.com/64748758/199672765-806c8c09-8409-4c32-aa44-273c07b97190.png)
   * Разметка дефектов не совсем очевидна. Дефекты размечены с большим запасом

## Описание решения:

Одним из важных факторов решения является скорость работы модели. Потому что на один рулон металла приходится тысячи изображений, а темпы производства таковы, что на оценку качества всего рулона нет и 5 минут.
По логике, на производстве большая часть продукции будет без дефектов. Поэтому можно обучить классификатор, который сразу будет отбирать изображения без дефектов, а остальные изображения пропускать через модель сегментации.
При использовании двух моделей мы два раза производим извлечение признаков из изображения. Возможео, получится ускорить работы системы, если общую часть backbone.
Были рассмотрены такие варианты:
* две разные модели
* модель с двумя головами, которая возвращает предсказание классификации и сегментации
* две модели с общим backbone (если дефект классифицируется, в модель сегментации подаются извлеченный на предыдущем этапе признаки) 

## Метрики оценивания:
В качестве метрики принят коэффициент Дайса, а также время работы модели

## Результаты обучения моделей
Оценка качества решения производилась на отложенном датасете из исходных данных.

### Модель классификации

| model           |                    Parameters                    | Accuracy | 
|-----------------|:------------------------------------------------:|----------|
| EfficientNetB1  |                  resize 256x512                  | 0.895    |
| EfficientNetB1  |          resize 256x512, augmentations           | 0.921    |
| EfficientNetB1  | resize 256x512, augmentations,<br>normalize<br/> | 0.943    |
| EfficientNetB2  | resize 256x512, augmentations,<br>normalize<br/> | 0.955    |
| EfficientNetB2  | crop 256x320, augmentations,<br>normalize,<br/>  | 0.958    |

### Модель сегментации

| model                 |                             Parameters                              | DICE coef | 
|-----------------------|:-------------------------------------------------------------------:|-----------|
| Unet                  |            resize 256x512, augmentations, EfficientNetB1            | 0.639     |
| Unet с общим backbone |            resize 256x512, augmentations, EfficientNetB1            | 0.592     |
| Unet с двумя головами |            resize 256x512, augmentations, EfficientNetB1            | 0,621     |
| LinkNet               |        resize 256x512, augmentations,<br>EfficientNetB2<br/>        | 0.640     |
| LinkNet               | crop 256x320, augmentations,<br>EfficientNetB2, weights pixels<br/> | 0.731     |

### Время работы
Скорость работы поверялась на GPU Tesla K80 с 12 Гб памяти на выборке из 2500 изображений

Выборка без дефектов:
* Разные модели - 244 секунды
* С двумя головами - 275 секунд
* С общим backbone - 244 секунды

Выборка 50/50:
* Разные модели - 380 секунд
* С двумя головами - 300 секунд
* С общим backbone - 495 секунд
Исходя из предположения, что заготовок без дефектов будет намного больше, можно сделать вывод, что разные модели будут работать быстрее
## Результаты:

![image](https://user-images.githubusercontent.com/64748758/199672794-58061f15-7f32-4c91-af85-6409097acdd1.png)
![image](https://user-images.githubusercontent.com/64748758/199672804-3dd4c7b8-7adc-4dcb-a108-e2e92e399241.png)

# Инструкция установки и запуска

Необходимо установить зависимости:

    pip install -r requirements.txt

Подгрузить нужные файлы:

    python3 -m scr.load_weights.py  # веса моделей
    python3 -m scr.load_test.py     # данные для проверки работы модели
    python3 -m scr.load_train.py    # данные для обучения моделей

#### Запуск из командной строки
Проверить, что в папку data/test загрузились файлы и запустить скрипт

    python3 main.py 

Предсказанные маски дефектов сохраняются в папку outputs
#### Запуск через flask 
Запустить скрипт и выбрать файл из папки data/test

    python3 flask_run.py 

