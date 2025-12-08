# GradientBoostAI — обучение градиентного бустинга

Проект содержит полностью переработанные ноутбуки и финальный пайплайн на Gradient Boosting Regressor. CSV-файлы оставлены без изменений.

## Что сделано
- Модель для прогноза стоимости жилья переключена на градиентный бустинг и протестирована на holdout-выборке.
- Добавлен baseline RandomForest для сравнения метрик.
- В ноутбуке есть таблица метрик, график фактических vs прогнозов и barplot важности признаков.

## Файлы

| Путь | Назначение |
| --- | --- |
| 01_pandas_playground.ipynb | Быстрый разбор приёмов pandas на `employees.csv` и синтетическом наборе. |
| 02_visualization_and_linear_models.ipynb | Визуализации и простая линейная регрессия на `Housing.csv` и `rost_ves.csv`. |
| 03_gradient_boosting_house_prices.ipynb | Полный ML-пайплайн для `housing_az_sqm_azn.csv` с градиентным бустингом, метриками и графиками. |
| housing_az_sqm_azn.csv | Основной датасет цен в AZN (исходный). |
| Housing.csv, employees.csv, rost_ves.csv | Дополнительные датасеты для примеров. |

## Обучение Gradient Boosting (ноутбук `03_gradient_boosting_house_prices.ipynb`)

| Шаг | Блок кода | Что происходит |
| --- | --- | --- |
| 1 | Импорт библиотек и загрузка `housing_az_sqm_azn.csv` | Чтение данных, перемешивание для стабильности. |
| 2 | Разделение признаков/цели | `target = 'PriceAZN'`, `X = dataset.drop(target)`, `y = dataset[target]`. |
| 3 | Анализ признаков | Статистики по числам и кардинальность категорий. |
| 4 | Препроцессинг | `ColumnTransformer`: числовые — медианный imputer + `StandardScaler`; категориальные — частотный imputer + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`. |
| 5 | Модели | `RandomForestRegressor` (baseline) и `GradientBoostingRegressor` (основная). |
| 6 | Holdout-оценка | `train_test_split(test_size=0.2, random_state=13)` + расчёт MAE/RMSE/R2 для обеих моделей. |
| 7 | Кросс-валидация | `cross_val_score` по R2 для финального бустинга. |
| 8 | Пример прогнозов | Прогноз цены для новых объектов (`Bedrooms`, `Bathrooms`, `Sqm`, `City`). |
| 9 | Важность признаков | Топ признаков из baseline леса + горизонтальный barplot. |
| 10 | Валидационный график | Scatter фактических vs предсказанных цен для градиентного бустинга. |
| 11 | Таблица метрик | DataFrame с MAE/RMSE/R2 для обеих моделей. |

## Метрики (появятся после выполнения ноутбука)

| Модель | MAE | RMSE | R2 | CV R2 (mean ± std) |
| --- | --- | --- | --- | --- |
| RandomForestRegressor | выводится в ноутбуке | выводится в ноутбуке | выводится в ноутбуке | — |
| GradientBoostingRegressor | выводится в ноутбуке | выводится в ноутбуке | выводится в ноутбуке | печатается после `cross_val_score` |

## Графики

| Где | Назначение |
| --- | --- |
| `02_visualization_and_linear_models.ipynb` | Гистограммы/боксплоты/регрессии для `Housing.csv`; линия регрессии для `rost_ves.csv`. |
| `03_gradient_boosting_house_prices.ipynb` | Scatter фактических vs прогнозов (Gradient Boosting) и barplot важности признаков (RandomForest). |

## Как воспроизвести обучение

1. Подготовить окружение с Python 3 и библиотеками `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, `numpy`.
2. Открыть `03_gradient_boosting_house_prices.ipynb` в Jupyter/VS Code.
3. Выполнить ячейки по порядку: загрузка данных → препроцессинг → обучение моделей → оценка → прогнозы → графики.
4. (Опционально) В `02_visualization_and_linear_models.ipynb` построить дополнительные графики для EDA.

После выполнения ноутбуков появятся таблицы с метриками и графики, отражающие качество и важность признаков модели.
