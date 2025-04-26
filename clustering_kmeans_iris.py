#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 22:05:43 2025

@author: andresllasessorice

Proyecto: Clustering con K-Means - Dataset IRIS
"""

# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import urllib.request

# Paso 1: Descargar el dataset automáticamente si no está localmente
ruta_destino = "./iris.csv"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
urllib.request.urlretrieve(url, ruta_destino)

# Paso 2: Cargar el archivo en un DataFrame
dataframe = pd.read_csv(
    ruta_destino,
    header=None,
    names=['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo', 'especies']
)

print("\nPrimeras filas del dataset:")
print(dataframe.head())

# Paso 3: Preparar las variables numéricas (X)
X = dataframe.iloc[:, :-1].values  # Todas las columnas menos la última

# Paso 4: Aplicar KMeans con 3 clusters
kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Paso 5: Graficar los clusters (usando las dos primeras variables)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', label='Centroides')
plt.xlabel('Longitud de sépalo')
plt.ylabel('Ancho de sépalo')
plt.title('Clusters de Iris dataset')
plt.legend()
plt.grid(True)
plt.show()

# Paso 6: Calcular SSM (inercia)
SSW = kmeans.inertia_
print(f"\nSSW (Suma de cuadrados dentro de los clusters): {SSW:.2f}")

# Paso 7: Calcular SSB (varianza entre clusters)
centro_global = np.mean(X, axis=0)
SSB = sum(
    [np.sum(labels == i) * np.sum((kmeans.cluster_centers_[i] - centro_global) ** 2) 
     for i in range(3)]
)
print(f"SSB (Suma de cuadrados entre clusters): {SSB:.2f}")
