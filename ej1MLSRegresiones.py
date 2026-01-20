
#Modelos de tipo regresiones 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score


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
    hidden_layer_sizes=(10,10),
    activation="relu",
    solver="adam",
    max_iter=500,
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
