## Структура датасета
Датасет, на котором я запускал обучение, имеет следующую структуру
```
2025_ipAdap_image
|
|__person
|____person1.jpg
|____person2.jpg
|____ ....
|____personN.jpg
|__next_person
|____next_person1.jpg
|____next_person2.jpg
|____ ....
|____next_personM.jpg
|__ ....
```

## Запуск моделей
Поскольку код использует библиотеку [Accelerate](https://huggingface.co/docs/accelerate/index), не забудьте настроить конфиг вашего окружения.
```bash
accelerate config
```

### IP-Adapter. Baseline
Следующий код позволит запустить генерацию изображений с помощью стандартного IP-Adapter. Запуск осуществляется командой
```bash
bash script.sh
```
или
```bash
python script.py --data_dir "/data/kazachkovda/2025_ipAdap_image/" --output_dir "../../figures" --json "../../data_preprocess/metadata.jsonl" --num_images 4
```
Перед этим вам необходимо установить все зависимости проекта и настроить пути к папке с данными. Описание структуры датасета можете найти в этом файле выше.

### IP-Adapter with Self-Attention
Следующий код позволит запустить генерацию изображений с помощью модифицированного IP-Adapter. Изменение заключается в том, что вместе с фотографией человека подается множество его фотографий с разных ракурсов. Эти ракурсы обрабатываются с помощью self-attention. Более подробно вы можете почитать в самой [работе](https://github.com/wolkendolf/2025-project-DiffModels/blob/main/docs/2025genavatars_main.pdf).

Флаг `images_number` означает количество ракурсов. Его вы можете установить как минимальное количество ракурсов среди имеющихся фотографий людей.
```bash
bash train_script.sh
```

### IP-Adapter with Latent Representaion
Следующий код позволит запустить генерацию изображений с помощью модифицированного IP-Adapter.