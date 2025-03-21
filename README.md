# XAI и кредитный скоринг на датасете кредитных карт

Данный репозиторий содержит полный рабочий пример интеграции методов **Explainable AI (XAI)** в задачу кредитного скоринга. Здесь вы найдёте скрипт на Python для обучения простой нейронной сети (MLP), а также расчёт и визуализацию объяснений при помощи **SHAP** (значения Шэпли).

---

## Содержимое репозитория

- **data.csv**  
  Файл с данными о клиентах (датасет кредитных карт). Включает признаки: лимит, история платежей, суммы задолженности и др.
- **script.py**  
  Основной скрипт, выполняющий следующие шаги:
  1. Загрузка и предобработка данных (удаление неиспользуемых столбцов, масштабирование).
  2. Обучение модели MLP (Keras, TensorFlow).
  3. Оценка точности и F1-меры. Вывод матрицы ошибок.
  4. Расчёт значений SHAP (KernelExplainer).
  5. Построение графиков Beeswarm и Bar для интерпретации модели.
- **Figure_1.png**  
  Условная иллюстрация (пример схемы, или любой рисунок, связанный со статьёй).
- **shap_beeswarm_rus.png**  
  График Beeswarm со значениями SHAP (подписи на русском).
- **shap_bar_rus.png**  
  Столбчатая диаграмма важности признаков по SHAP (также с русскими подписями).

---

## Быстрый старт

1. **Установите необходимые библиотеки**:

    ```bash
    pip install -r requirements.txt
    ```
    Возможный список библиотек: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `tensorflow`, `shap`.

2. **Запустите скрипт**:

    ```bash
    python script.py
    ```
    - Произойдёт чтение и предобработка данных из `data.csv`.
    - Обучится модель (MLP) на 20 эпохах.
    - Выведутся метрики на тестовой выборке (Accuracy, F1-score).
    - Сформируются и сохранятся в файлах PNG-графики (матрица ошибок, SHAP beeswarm, SHAP bar).

3. **Рассмотрите результаты**:
   - Метрики и отчёт по классификации отобразятся в консоли.
   - Все графики (включая матрицу ошибок) откроются при помощи `matplotlib`, а также сохранятся рядом со скриптом.

---

## Рекомендации по настройке

- При работе с данным скриптом следите за соответствием версий библиотек (особенно `tensorflow` и `shap`), чтобы избежать конфликтов.
- Для более точных или более быстрых результатов возможно изменение архитектуры MLP, количества эпох или размера батча (параметры задаются в `script.py`).

---

## Лицензия

Вы можете свободно использовать данный код и файлы датасета в некоммерческих исследованиях и экспериментах. Для иных целей уточняйте условия исходной лицензии датасета UCI.

---

## Контактная информация

- **Автор**: Andrei Chmelev
- **Email**: an.chmelev@gmail.com

Если у вас есть вопросы или предложения по улучшению репозитория, открывайте [issue](https://github.com/username/название-репо/issues) или присылайте pull request.
```