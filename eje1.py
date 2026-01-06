#IA DEBIL

#Paso 1. importacion de Librerias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Paso 2. Obtener el dataset
datos= load_iris()
entradas= datos.data 
salidas=datos.target

#Paso 3. Division de datos para entrenamiento y testeo
entradas_entrenamiento, entradas_testeo, salidas_entrenamiento, salidas_testeo=train_test_split(entradas,salidas, test_size=0.2)

#Paso 4. Creacion y entrenamiento del modelo
modelo=KNeighborsClassifier(n_neighbors=3)
modelo.fit(entradas_entrenamiento,salidas_entrenamiento)

#Paso 5.Prueba del modelo (prediccion)
predicciones=modelo.predict(entradas_testeo)
precision=accuracy_score(salidas_testeo,predicciones)

print(f"Precision del Modelo")
print(precision)

