#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

# Cargar el dataset
df = pd.read_csv('COVID19C1.csv')

# Mostrar estadísticas generales del dataset
#print(tabulate(df.describe(), headers='keys', tablefmt='psql'))

# Verificar si hay valores faltantes
#print("\nValores faltantes en cada columna:")
#print(df.isnull().sum())

# Verificar el desequilibrio de clases
print("\nDistribución de clases en 'COVID19_Severity':")
print(df['COVID19_Severity'].value_counts(normalize=True))

# Separar las características y la variable objetivo
X = df.drop('COVID19_Severity', axis=1)
y = df['COVID19_Severity']

# Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Estandarizar las características
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Inicializar los modelos con pesos de clase
tree = DecisionTreeClassifier(class_weight='balanced', random_state=42)
forest = RandomForestClassifier(class_weight='balanced', random_state=42)
boosting = GradientBoostingClassifier(random_state=42)
log_reg = LogisticRegression(class_weight='balanced', random_state=42)
svm = SVC(class_weight='balanced', random_state=42)
nn = MLPClassifier(max_iter=500, random_state=42)

# Entrenar y evaluar los modelos
models = {'Decision Tree': tree, 'Random Forest': forest, 'Gradient Boosting': boosting,
          'Logistic Regression': log_reg, 'SVM': svm, 'Neural Network': nn}
scores = {}

for model_name, model in models.items():
    # Entrenar el modelo
    model.fit(X_train, y_train)
    # Hacer predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    # Evaluar el modelo
    scores[model_name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

# Imprimir los puntajes de los modelos
for model_name, score in scores.items():
    print("\nModelo:", model_name)
    print("Exactitud:", score['accuracy'])
    print("Reporte de clasificación:\n", score['classification_report'])
    print("Matriz de confusión:\n", score['confusion_matrix'])

# Inicializar un objeto StratifiedKFold
strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Realizar la validación cruzada en todos los modelos y recoger los resultados
cross_val_scores_all = {}

for model_name, model in models.items():
    # Realizar la validación cruzada
    cross_val_scores = cross_val_score(model, X, y, cv=strat_k_fold, scoring='accuracy')
    # Recoger los resultados
    cross_val_scores_all[model_name] = cross_val_scores

# Calcular la correlación absoluta de cada característica con la variable objetivo
correlations = df.corr()['COVID19_Severity'].apply(np.abs).sort_values(ascending=False)

# Obtener las 10 características más correlacionadas
top_features = correlations.index[1:11]  # Comenzamos desde 1 para excluir a 'COVID19_Severity' en sí

# Imprimir las 10 características más correlacionadas
print("\nLas 10 características más correlacionadas con 'COVID19_Severity':")
print(top_features)

# Dibujar un mapa de calor de las correlaciones
plt.figure(figsize=(12, 10))
sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f")
plt.show()

# Inicializar un diccionario para almacenar los puntajes de validación cruzada de cada modelo
cv_scores_features = {}

# Para cada número de características de 10 a 1
for num_features in range(10, 0, -1):
    # Seleccionar las num_features más altas
    selected_features = top_features[:num_features]
    # Seleccionar las columnas correspondientes del dataset
    X_selected = X[selected_features]

    # Split the data
    X_train_selected, X_test_selected, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize the features
    X_train_selected = sc.fit_transform(X_train_selected)
    X_test_selected = sc.transform(X_test_selected)

    # Train and evaluate the model
    forest.fit(X_train_selected, y_train)
    y_pred = forest.predict(X_test_selected)

    # Print the metrics
    print("\nModel trained on top", num_features, "features:")
    print("Features used:", ", ".join(selected_features))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    #print("Classification report:\n", classification_report(y_test, y_pred))
    # print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Perform cross validation
    cv_scores = cross_val_score(forest, X_selected, y, cv=strat_k_fold, scoring='accuracy')
    # Almacenar el puntaje medio de validación cruzada
    cv_scores_features[num_features] = cv_scores.mean()

# Imprimir los puntajes de validación cruzada para cada número de características
print("\nPuntajes de validación cruzada para cada número de características:")
for num_features, cv_score in cv_scores_features.items():
    print("Número de características:", num_features, ", Puntaje de validación cruzada:", cv_score)

