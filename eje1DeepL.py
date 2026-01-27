#Importacion de librerias
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

#dataset
data=pd.DataFrame({
   "Mes":[1,2,3,4,5,6,7,8,9,10,11,12],
   "Publicidad":[200,150,100,250,268,286,304,322,340,358,376,394],
    "Precio":[20,15,10,15,20,25,30,35,40,45,50,55],
    "Ventas":[150,160,170,180,190,200,210,220,230,240,250,260],  
})

print(data)

x=data[["Mes","Precio","Publicidad"]]
y=data["Ventas"]

escalamiento=StandardScaler()
xEscalado=escalamiento.fit_transform(x)

#Division de datos
xEntranamiento,xTesteo,yEntrenamiento,yTesteo=train_test_split(
    xEscalado,
    y,
    test_size=0.20,
    random_state=42
)

#Creacion del modelo deep learning (metodo para agregar capas ocultas add)
modelo=Sequential()

modelo.add(Dense(32,activation="relu",input_shape=(3,)))

#Capas ocultas
modelo.add(Dense(16,activation="relu"))
modelo.add(Dense(8,activation="relu"))

#Capa salida
modelo.add(Dense(1))

#Compilar el modelo
modelo.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

#entrenamiento
entrenamiento=modelo.fit(
    xEntranamiento,
    yEntrenamiento,
    epochs=100,
    validation_split=0.2,
    batch_size=8,
)

#Cantidad de epocas segun el dataset
#datasets Pequeños(10 -1000 registros) = 100-500 epocas 
#datasets Medianos(1000 - 100000 registros) = 50 - 300 epocas
#datasets Grandes (100000 - millones de registros)= 10 - 100 epocas

#caso extra dataset inmensos (billones de registros) 500-1000

graficaLOSS=modelo.loss_curve_
#loos vs epochs
plt.figure()
plt.plot(range(1,len(graficaLOSS)+1),graficaLOSS,)
plt.show()

#print(graficaLOSS)