#MODELO SUPERVISADOS - CLASIFICATORIOS
# REGRESIONES LOGISTICAS

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#['el kit de machine learning']
#[['el'],[' '],['kit'],[de],['machine'],['learning']]

#['La redes neuronales (nodos entrantes, nodos ocultos, nodos salientes)']

#[['('],[',']]

#Definicion de las categorias
categorias={
    0:"Deportes",
    1:"Cultural",
    2:"Politico",
    3:"Economico",
    4:"Policial",
    5:"Ciencia",
}

#Data para procesamiento

noticias=[
    "La Selección nacional concentrará en Santa Cruz con vistas a los tres encuentros de preparación", #deportivo
    "Objeto interestelar 3I/ATLAS muestra señales que ningún otro cometa debería tener", #cultural
    "El gobierno nacional exige 'por escrito' las observaciones de la COB al D.S. 5503", #Politico,
    "Hallan sin vida a la niña de ocho años reportada como desaparecida en La Guardia", #Policial
    "Fiscalía solicita sello rojo a Interpol en contra de Armin Dorgathen, el ex presidente de YPFB", #Policial
    "Mamani ya es boliviano y se pone a disposición de Villegas en la Verde", #deportivo
    "Bolivia ajusta estrategia de comercio exterior con base en cinco líneas de trabajo", #Economico
    "Productores alertan pérdidas millonarias y mercados en riesgo", #Economico
    "Diálogo entra en cuarto intermedio y Gobierno pide propuestas claras a la COB", #Politico
    "Aprehenden al exministro de Obras Públicas, Édgar Montaño", #Politico
]

etiquetas=[
    0,1,2,4,4,0,3,3,2,2
]
#Conversion de texto a numeros
vectorizacion=CountVectorizer()
x=vectorizacion.fit_transform(noticias)
y=etiquetas

#Separacion de los datos
#80% entrenamiento, 20% pruebas
xEntrenamiento,xPruebas, yEntrenamiento, yPruebas=train_test_split(x,y,test_size=0.3)

#Crear modelo 
modelo=LogisticRegression(
    max_iter=1000,
    #multi_class="auto"
)

#Entrenamiento
modelo.fit(xEntrenamiento,yEntrenamiento)

#Prediccion con datos de prueba
predicciones=modelo.predict(xPruebas)
predicciones2=modelo.predict_proba(xPruebas)
precision=accuracy_score(yPruebas,predicciones)

print("Precision del modelo")
print(precision*100,'%')

#Verificacion detallada
for i in range(len(predicciones)):
    real=categorias[yPruebas[i]]
    valorPredicho=categorias[predicciones[i]]
    print(f"Real:{real} -> Predicción:{valorPredicho}")


#prediccion a dato nuevo
nuevaNoticia=["La tensión en los bloqueos aumenta; son 56 puntos y los pasajeros demandan libre tránsito"]
vectorizarNoticia=vectorizacion.transform(nuevaNoticia)
prediccionNuevo=modelo.predict(vectorizarNoticia)[0]
probabilidades=modelo.predict_proba(vectorizarNoticia)[0]

print("Clasificacion de la nueva noticia")
print("Noticia: ",nuevaNoticia[0])
print("Categoria Predicha: ",categorias[prediccionNuevo])

print("Probabilidades por categoria")
for i, prob in enumerate(probabilidades):
    print(f"{categorias[i]}: {prob*100}%")



#5 Predicciones 







