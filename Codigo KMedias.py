# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:47:04 2025

@author: danir
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from ucimlrepo import fetch_ucirepo

# Cargar dataset
wine = fetch_ucirepo(id=109)

# Ver los nombres reales de las columnas

# Seleccionar solo dos columnas existentes (ajustar según nombres reales)
columnas = ["Alcohol", "Malicacid"]  # Asegúrate de que estos nombres coincidan exactamente

X = wine.data.features[columnas]

# Graficar datos sin clasificar
plt.figure(figsize=(6, 5), dpi=100)
plt.scatter(X["Alcohol"], X["Malicacid"], color="green", alpha=0.5, s=80)
plt.title("Vinos", fontsize=15)
plt.xlabel("Alcohol", fontsize=12)
plt.ylabel("Ácidez", fontsize=12)
plt.show()
# Normalizar datos
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Aplicar KMeans con 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
X_scaled["cluster"] = kmeans.fit_predict(X_scaled)

# Graficar clusters
plt.figure(figsize=(6, 5), dpi=100)
colores = ["red", "blue", "orange"]

for cluster in range(kmeans.n_clusters):
    plt.scatter(X_scaled[X_scaled["cluster"] == cluster]["Alcohol"], 
                X_scaled[X_scaled["cluster"] == cluster]["Malicacid"],
                marker="o", s=180, color=colores[cluster], alpha=0.5, label=f"Grupo {cluster}")

    plt.scatter(kmeans.cluster_centers_[cluster][0], 
                kmeans.cluster_centers_[cluster][1], 
                marker="P", s=280, color=colores[cluster], edgecolors="black")

plt.title("Clasificación de Vinos (Alcohol vs. Acidez)", fontsize=15)
plt.xlabel("Alcohol (Normalizado)", fontsize=12)
plt.ylabel("Acidez(Normalizado)", fontsize=12)
plt.text(1.02, 0.05, f"K = {kmeans.n_clusters}", fontsize=12, transform=plt.gca().transAxes)
plt.legend()
plt.show()

# Eliminar columna 'cluster' del DataFrame
X_scaled.drop(columns=["cluster"], inplace=True)


