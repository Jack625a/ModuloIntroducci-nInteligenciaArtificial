
#Modelos de tipo regresiones 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


#Dataset
data=pd.DataFrame({
    "Producto":["Cafe","Juego","Leche","Cafe","Juego","Leche","Cafe","Juego","Leche","Cafe","Juego","Leche"],
    "Mes":[1,2,3,4,5,6,7,8,9,10,11,12],
    "Publicidad":[200,150,100,250,268,286,304,322,340,358,376,394],
    "Precio":[20,15,10,15,20,25,30,35,40,45,50,55],
    "Ventas":[150,160,170,180,190,200,210,220,230,240,250,260],
    "Total":[3000,2400,1700,2700,3800,5000,6300,7700,9200,10800,12500,14300],
    "Ganancia":[2800,2250,1600,2450,3532,4714,5996,7378,8860,10442,12124,13906]
})

print(data)

#preprocesamiento
#conversion de la columna Producto a representacion numerica
productosConvertidos=pd.get_dummies(data["Producto"],prefix="Producto")

#unir los datosLimpios al dataframe
x=pd.concat([productosConvertidos,data[["Mes","Publicidad","Precio","Total","Ganancia"]]],axis=1)
y=data["Ventas"]

escalamiento=StandardScaler()
xEscalado=escalamiento.fit_transform(x)

#Division de entrenamiento / testeo
xEntrenamiento,xPrueba,yEntrenamiento,yPrueba=train_test_split(xEscalado,y,test_size=0.20,random_state=42)

#crear el modelo (PERCEPTRON MULTICAPA REGRESSIONES)
modelo=MLPRegressor(
    hidden_layer_sizes=(30,20,10),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

#entrenar modelo
modelo.fit(xEntrenamiento,yEntrenamiento)

predicciones=modelo.predict(xPrueba)
mcuadratica=mean_squared_error(yPrueba,predicciones)
coeficienteDeterminacion=r2_score(yPrueba,predicciones)
#error cuadratico medio
#cOEFICIENTE DE DETEMINACION 
print("Error cuadratico medio: ",mcuadratica)
print("Coeficiencete de Determinacion: ",coeficienteDeterminacion)

#Visualizar los datos 
#cremiento de las ventas 
plt.figure()
plt.plot(data["Mes"],data["Ventas"])
plt.xlabel("Mes")
plt.ylabel("Ventas")
plt.title("Evolucion de ventas por mes")
plt.show()

#Publicidad /Ventas
plt.figure()
plt.scatter(data["Publicidad"],data["Ventas"])
plt.xlabel("Publicidad")
plt.ylabel("Ventas")
plt.title("Relacion entre publicidad y ventas")
plt.show()

plt.figure()
plt.scatter(data["Precio"],data["Ventas"])
plt.xlabel("Precio")
plt.ylabel("Ventas")
plt.title("Relacion entre Precio y Ventas")
plt.show()



#Grafico de valores reales historico vs predicciones realizadas por el modelo
plt.figure()
plt.scatter(yPrueba,predicciones)
plt.xlabel("Ventas Reales")
plt.ylabel("Ventas Predichas")
plt.title("Ventas Reales vs Ventas Predichas")
plt.show()

#Visualizar el erro del modelo
errores=yPrueba-predicciones
plt.figure()
plt.scatter(range(len(errores)),errores)
plt.axhline(0)
plt.xlabel("Datos")
plt.ylabel("Error")
plt.title("Errores del modelo")
plt.show()

#Eje x Publicidad
#Eje y Precio
#Eje z Ventas
figura3d=plt.figure()
grafica=figura3d.add_subplot(111,projection="3d")
grafica.scatter(data["Publicidad"],data["Precio"],data["Ventas"])

grafica.set_xlabel("Publicidad")
grafica.set_ylabel("Precio")
grafica.set_zlabel("Ventas")
grafica.set_title("Publicidad vs Precio vs Ventas")

plt.show()
