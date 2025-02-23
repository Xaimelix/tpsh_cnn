# VGG Image Classifier Flask App #

Веб-приложение Flask для классификации изображений с использованием модели VGG16, предварительно обученной на ImageNet.

## Установка и запуск ## 
Установите [Python 3.12](https://www.python.org/downloads/release/python-3120/) 

Создайте виртуальное окружение
  ```
  python -m venv venv
  source venv/bin/activate  # Linux/macOS
  venv\Scripts\activate  # Windows
  ```

Для установки библиотек загрузите их из requirements.txt
   ```
   pip install -r requirements.txt
```

## Технологии

Проект разработан с использованием:

* Python
* Flask (веб-фреймворк)
* PyTorch (библиотека глубокого обучения)
* torchvision (PyTorch Vision - для моделей и преобразований изображений)
* Pillow (PIL - для работы с изображениями)
