# Full-length human image generation using diffusion models.

**Author:** Kazachkov Daniil <br>
**Supervisor:** Filatov Andrew

## Problem Statement
Наша главная цель - расширить область применения диффузионных моделей в генерации высококачественных изображений. Мы проверяем гипотезу о том, что подходы в работах [PuLID](https://github.com/ToTheBeginning/PuLID), [InstantID](https://instantid.github.io/), [IP-Adapter](https://ip-adapter.github.io/) распространяются не только на аватары лиц, но и на ростовые. Мы достигаем этого изменением датасета и энкодера, принимающего на вход референсную картинку пользователя.
Отдельной частью эксперимента является попытка подобрать лучший метод для выделения тела человека (BodyID) с картинки. 

## Train baseline model
Для этого установите все зависимости из [IP-Adapter](https://ip-adapter.github.io/), замените `tutorial_train_plus.py` на соответствующий ему файл `./model_train/tutorial_train_plus.py`. Запуск скрипта осуществляется выполнением скрипта `train.sh`, который предварительно надо перенести в директорию `IP-Adapter`. 

Результаты тренировки модели можно найти в ноутбуке 


## Citation
TODO
