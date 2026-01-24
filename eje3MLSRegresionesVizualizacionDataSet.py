
#Modelos de tipo regresiones 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


#Dataset
data=pd.read_csv("datasetSimulado.csv")
#print(data.head())
#print(data.info())

#Limpieza extra
data=data.dropna()
data=data.sample(frac=1,random_state=42)



#preprocesamiento
#conversion de la columna Producto a representacion numerica
productosConvertidos=pd.get_dummies(data["Producto"],prefix="Producto")

#unir los datosLimpios al dataframe
x=pd.concat([productosConvertidos,data[["Mes","Publicidad","Precio","Total","Ganancia"]]],axis=1)
y=data["Ventas"]

escalamiento=StandardScaler()
xEscalado=escalamiento.fit_transform(x)

#Division de entrenamiento / testeo
xEntrenamiento,xPrueba,yEntrenamiento,yPrueba=train_test_split(xEscalado,y,test_size=0.30,random_state=42)

#crear el modelo (PERCEPTRON MULTICAPA REGRESSIONES)
modelo=MLPRegressor(
    hidden_layer_sizes=(30,20,10),
    activation="relu",
    solver="adam",
    max_iter=2500,
    random_state=42,
    early_stopping=True, #ENTRENAMIENTO OPTIMO
    #validation_fraction=0.15

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

