#MODELO SUPERVISADOS - CLASIFICATORIOS
# REGRESIONES LOGISTICAS
from sklearn.linear_model import LogisticRegression
import numpy as np

#DATOS DE PROCESAMIENTO - cantidad de horas de estudio
x=np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y=np.array([0,0,0,1,1,1,1,1]) #1 aprobado - 0 reprobado

#Creacion del modelo
modelo=LogisticRegression()

#Entrenar el modelo
modelo.fit(x,y)

#Prediccion (ej1. Probabilidad de aprobar estudiando 4 horas)
prediccionEstudiante=np.array([[9]])
probabilidad=modelo.predict_proba(prediccionEstudiante)[0][1]

prediccionFinal=modelo.predict(prediccionEstudiante)[0]

print("Probabilidad de aprobar", prediccionFinal)




#MODELO SUPERVISADOS - REGRESIONES


#MODELO NO SUPERVISADOR - CLUSTER