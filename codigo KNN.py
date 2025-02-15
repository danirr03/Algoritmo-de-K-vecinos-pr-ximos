# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 11:51:30 2025

@author: danir
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# Cargar el dataset
semillas = pd.read_csv("seeds_dataset.txt", sep="\s+", header=None)

# Asignar nombres de columnas
semillas.columns = ["Area", "Perimetro", "Compacidad", "Longitud", "Ancho", "Asimetria", "LongitudRanura", "Clase"]

# Filtrar las tres clases
kama = semillas[semillas["Clase"] == 1]
rosa = semillas[semillas["Clase"] == 2]
canadian = semillas[semillas["Clase"] == 3]

# Visualizar datos (usamos Asimetría y Perímetro)
plt.figure(figsize=(6, 5))
plt.scatter(kama["Asimetria"], kama["Perimetro"], marker="o", s=50, color="skyblue", label="Kama (Clase: 1)")
plt.scatter(rosa["Asimetria"], rosa["Perimetro"], marker="s", s=50, color="red", label="Rosa (Clase: 2)")
plt.scatter(canadian["Asimetria"], canadian["Perimetro"], marker="*", s=50, color="green", label="Canadian (Clase: 3)")

plt.ylabel("Perímetro de la semilla")
plt.xlabel("Asimetría de la semilla")
plt.legend(bbox_to_anchor=(1, 0.5)) 
plt.title("Distribución de las Clases")
plt.show()

# Seleccionar características para clasificación
datos = semillas[["Asimetria", "Perimetro"]]
clase = semillas["Clase"]

# Normalizar los datos
escalador = preprocessing.MinMaxScaler()
datos = escalador.fit_transform(datos)

# Crear y entrenar el clasificador KNN con 15 vecinos
clasificador = KNeighborsClassifier(n_neighbors=14)
clasificador.fit(datos, clase)

# Nueva semilla a clasificar (ejemplo con valores promedio del dataset)
asimetria_nueva = 4.0
perimetro_nueva = 13.5

# Escalar los datos del nuevo dato a predecir
nueva_semilla = escalador.transform([[asimetria_nueva, perimetro_nueva]])

# Predicción de la clase
print("Clase predicha:", clasificador.predict(nueva_semilla))
print("Probabilidades por clase:", clasificador.predict_proba(nueva_semilla))

# Graficar la nueva semilla en el conjunto de datos
plt.figure(figsize=(6, 5))
plt.scatter(kama["Asimetria"], kama["Perimetro"], marker="o", s=50, color="skyblue", label="Kama (Clase: 1)")
plt.scatter(rosa["Asimetria"], rosa["Perimetro"], marker="s", s=50, color="red", label="Rosa (Clase: 2)")
plt.scatter(canadian["Asimetria"], canadian["Perimetro"], marker="*", s=50, color="green", label="Canadian (Clase: 3)")
plt.scatter(asimetria_nueva, perimetro_nueva, marker="P", s=100, color="black", label="Nueva semilla")

plt.ylabel("Perímetro de la semilla")
plt.xlabel("Asimetría de la semilla")
plt.legend(bbox_to_anchor=(1, 0.5))
plt.title("Nueva semilla clasificada")
plt.show()

# --- NUEVA GRÁFICA: REGIONES DE PREDICCIÓN ---

# Crear una malla de puntos para graficar regiones de decisión
x_min, x_max = datos[:, 0].min() - 0.1, datos[:, 0].max() + 0.1
y_min, y_max = datos[:, 1].min() - 0.1, datos[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Predecir la clase en cada punto de la malla
Z = clasificador.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar las regiones de decisión
plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

# Graficar los datos originales con las semillas de diferentes clases
plt.scatter(datos[clase == 1, 0], datos[clase == 1, 1], marker="o", s=50, color="skyblue", label="Kama (Clase: 1)")
plt.scatter(datos[clase == 2, 0], datos[clase == 2, 1], marker="s", s=50, color="red", label="Rosa (Clase: 2)")
plt.scatter(datos[clase == 3, 0], datos[clase == 3, 1], marker="*", s=50, color="green", label="Canadian (Clase: 3)")

# Graficar la nueva semilla
plt.scatter(nueva_semilla[0, 0], nueva_semilla[0, 1], marker="P", s=100, color="black", label="Nueva semilla")

plt.ylabel("Perímetro de la semilla (normalizado)")
plt.xlabel("Asimetría de la semilla (normalizado)")
plt.title("Regiones de Predicción con KNN")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()










