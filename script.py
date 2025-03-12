# -------------------------------------------------------------
# Полный финальный скрипт на Python для эксперимента (XAI)
# с датасетом кредитных карт. Все части кода приведены целиком!
# -------------------------------------------------------------
# Скрипт:
#  1. Загружает CSV-файл, убирает лишний заголовок (header=1).
#  2. Предобрабатывает данные (drop ID, разделяет на X и y).
#  3. Делит на обучающую/тестовую выборки, масштабирует признаки.
#  4. Обучает MLP-модель (Keras).
#  5. Оценивает точность и F1-score.
#  6. Делает SHAP-анализ (KernelExplainer).
#  7. Сохраняет графики SHAP с русскими подписями и экспортирует метрики в .md.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models

import shap

# -----------------------------------------------------
# 1. Загрузка данных и первичный обзор DataFrame
# -----------------------------------------------------
df = pd.read_csv("data.csv", header=1, encoding="utf-8")

print("Первые 5 строк датасета:")
print(df.head())

print("\nИнформация о DataFrame:")
print(df.info())

# ------------------------------------------------------------
# 2. Предобработка (удаление ID, деление на X и y)
# ------------------------------------------------------------
if 'ID' in df.columns:
    df.drop('ID', axis=1, inplace=True)

target_col = 'Y'
if 'Y' not in df.columns:
    target_col = 'default payment next month'

df.dropna(axis=0, inplace=True)

X = df.drop(columns=[target_col])
y = df[target_col]

# Разделяем данные на train/val и test.
# stratify=y, чтобы сохранить пропорции классов
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Масштабируем числовые признаки
scaler = StandardScaler()
X_trainval_scaled = scaler.fit_transform(X_trainval)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------
# 3. Построение и обучение MLP-модели (Keras)
# ------------------------------------------------
model = models.Sequential()
model.add(layers.Input(shape=(X_trainval_scaled.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_trainval_scaled,
    y_trainval,
    epochs=20,
    batch_size=256,
    validation_split=0.2,  # 80% на обучение, 20% на валидацию
    verbose=1
)

# -------------------------------------------------
# 4. Оценка качества на тестовой выборке (test)
# -------------------------------------------------
y_pred_test = model.predict(X_test_scaled)
y_pred_labels = (y_pred_test >= 0.5).astype(int)

acc_test = accuracy_score(y_test, y_pred_labels)
f1_test = f1_score(y_test, y_pred_labels, average='binary')

print(f"\nТочность (accuracy) на тестовой выборке: {acc_test:.4f}")
print(f"F1-мера (f1-score) на тестовой выборке: {f1_test:.4f}")

print("\nПодробный отчёт (classification_report):")
print(classification_report(y_test, y_pred_labels))

cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap='Blues')
plt.title("Матрица ошибок (Confusion Matrix)")
plt.colorbar()
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

# --------------------------------
# 5. Анализ с помощью SHAP
# --------------------------------
explainer_sample_size = 200
X_explain = X_test_scaled[:explainer_sample_size]

def model_predict(data):
    return model.predict(data).flatten()

explainer = shap.KernelExplainer(model_predict, data=X_explain)
num_explain = 50

# Вычисляем shap_values для первых 50 объектов
shap_values = explainer.shap_values(X_explain[:num_explain], nsamples=100)

# (a) Beeswarm-график (dot), русские подписи
shap.summary_plot(
    shap_values,
    features=X_explain[:num_explain],
    feature_names=X.columns.to_list(),
    plot_type='dot',
    show=False  # отключаем автопоказ
)
plt.gcf().axes[-1].set_ylabel("Значение признака (меньше - синий, больше - красный)")
plt.gca().set_xlabel("Значение SHAP (влияние на выход модели)")
plt.title("Beeswarm-график SHAP")
plt.savefig("shap_beeswarm_rus.png", dpi=120, bbox_inches='tight')
plt.show()

# (b) Bar-график средних абсолютных значений SHAP
shap.summary_plot(
    shap_values,
    features=X_explain[:num_explain],
    feature_names=X.columns.to_list(),
    plot_type='bar',
    show=False
)
plt.gca().set_xlabel("Среднее абсолютное значение SHAP\n(средний вклад в результат модели)")
plt.title("Столбчатая диаграмма важности признаков (SHAP)")
plt.savefig("shap_bar_rus.png", dpi=120, bbox_inches='tight')
plt.show()

# ------------------------------------------------------
# 6. Итоговая сводная таблица результатов (в MD-файл)
# ------------------------------------------------------
results_df = pd.DataFrame({
    'Метрика': ['Точность (Accuracy)', 'F1-мера (F1-score)'],
    'Значение': [acc_test, f1_test]
})

print("\nИтоговая сводная таблица с результатами (test):")
print(results_df)

