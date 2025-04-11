# Data processing

Порядок запуска:
1. check_watermarks.py (дополнительно удалим повержденные файлы)
2. check_number_faces.py
3. (optional) check_count_imgs.py (можно поставить требование минимального количества изображений в папке. Все папки ниже данного порога будут удалены)
4. generate_metadata_gpuVersion.py

## Check watermarks

### Desription
Решение основывается на проекте [watermark-detection](https://github.com/boomb0om/watermark-detection), откуда необходимо установить зависимости.

Код способен обрабатывать битые изображения.

### Run
После клонирования репозитория [watermark-detection](https://github.com/boomb0om/watermark-detection), необходимо прописать в `data/check_watermarks.py` путь к модели:

```python
sys.path.append(
    "/home/jovyan/nkiselev/kazachkovda/2025-project-DiffModels/watermark-detection"
)
```

Также не забудьте указать путь к датасету и устройство.
```python
BASE_DATASET_PATH = (
    "/path/to/dataset"
)
device = 'cuda:0'
```

Запуск обработчика
```bash
cd data
python check_watermarks.py
```


## Face validation

### Description
Проверка фотографий на наличие лишь одного лица на изображении. Основывывается на модели `YOLOv11n-face-detection`.

### Run
Не забудьте указать путь к датасету:
```python
BASE_DATASET_PATH = "/path/to/dataset"
```

После установки необходимых зависимостей:
```bash
cd data 
python check_number_faces.py
```


## Desriptive generation

### Description
В основе лежит модель `Qwen2.5-VL-7B-Instruct`.

### Run
После установки необходимых зависимостей:

```bash
cd data 
python generate_metadata_gpuVersion
```

