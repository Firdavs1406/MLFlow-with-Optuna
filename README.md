# MLFlow with Optuna

## Запуск проекта с интеграцией Optuna и MLFlow

Этот проект включает процесс оптимизации гиперпараметров с помощью Optuna, обучение модели CatBoost и логирование результатов в MLFlow.

---

## 1. Установка зависимостей

Перед запуском убедитесь, что все необходимые библиотеки установлены. Выполните команду:

```bash
pip install pandas catboost sklearn optuna mlflow shap matplotlib
```

---

## 2. Запуск MLFlow Tracking Server

Для логирования параметров, метрик и моделей требуется запустить MLFlow Tracking Server. Выполните команду:

```bash
mlflow ui
```

Сервер будет доступен по адресу: [http://127.0.0.1:5000](http://127.0.0.1:5000). Убедитесь, что порт 5000 свободен.

---

## 3. Организация файлов

Структура проекта выглядит следующим образом

```
project/
├── data/
│   └── data.csv               # Исходный датасет
├── preprocessing/
│   └── prepare_data.py        # Скрипт подготовки данных
├── utils/
│   └── utils.py               # Утилиты для визуализации и оценки модели
├── model/
│   └── train.py               # Скрипт для обучения модели
├── main.py                    # Главный файл (код с Optuna)
```

### Описание папок и файлов:
- **data/data.csv**: файл с данными.
- **preprocessing/prepare_data.py**: содержит функции для обработки данных и подготовки пулов CatBoost.
- **utils/utils.py**: утилиты для оценки модели и построения графиков.
- **model/train.py**: Содержит код, для обучения модели
- **main.py**: главный скрипт для выполнения обучения и оптимизации.

---

## 4. Запуск скрипта

Выполните команду в корневой папке проекта:

```bash
python main.py
```

---

## 5. Результаты работы

### На терминале:
- Отобразятся лучшие гиперпараметры, метрики модели и прогресс оптимизации.
- Пример вывода:
  ```
  Best parameters: {'iterations': 500, 'depth': 6, 'learning_rate': 0.05}
  Test Accuracy: 0.89
  ```

### В MLFlow UI:
1. Перейдите в [http://127.0.0.1:5000](http://127.0.0.1:5000).
2. Найдите эксперимент `optuna_catboost_experiment`.
3. Внутри эксперимента будут сохранены:
   - Логи гиперпараметров (`learning_rate`, `iterations`, `depth`).
   - Метрики (`accuracy`).
   - Графики Optuna:
     - Параллельные координаты (`parallel_coordinates.html`).
     - История оптимизации (`optimization_history.html`).
   - Важность признаков (`feature_importance.png`).
   - SHAP-график (`shap.png`).

---
