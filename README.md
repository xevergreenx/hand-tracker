# Hand Tracker — Gesture Recognition

Простой проект для распознавания жестов одной руки в реальном времени с веб-камеры.
Использует MediaPipe для извлечения ключевых точек и нейросеть на PyTorch для классификации жестов.

---

## Стек
```
Python 3.9
MediaPipe
OpenCV
PyTorch
NumPy
```
---

## Структура проекта
```
collect_data.py   # сбор датасета
train.py          # обучение модели
model.py          # архитектура сети
real_time.py      # инференс с камеры
gesture.pth       # обученные веса
data/             # датасет
pic/              # изображения
```
