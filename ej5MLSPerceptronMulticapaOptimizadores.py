#importacion libreria para preprocesamiento
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


nltk.download("stopwords")
nltk.download('punkt_tab')
#Data para procesamiento

noticias=[
    "La Selección Nacional concentrará en Santa Cruz con vistas a los tres encuentros de preparación", #deportivo
    "Objeto interestelar 3I/ATLAS muestra señales que ningún otro cometa debería tener", #cultural
    "El gobierno nacional exige 'por escrito' las observaciones de la COB al D.S. 5503", #Politico,
    "Hallan sin vida a la niña de ocho años reportada como desaparecida en La Guardia", #Policial
    "Fiscalía solicita sello rojo a Interpol en contra de Armin Dorgathen, el ex presidente de YPFB", #Policial
    "Mamani ya es boliviano y se pone a disposición de Villegas en la Verde", #deportivo
    "Bolivia ajusta estrategia de comercio exterior con base en cinco líneas de trabajo", #Economico
    "Productores alertan pérdidas millonarias y mercados en riesgo", #Economico
    "Diálogo entra en cuarto intermedio y Gobierno pide propuestas claras a la COB", #Politico
    "Aprehenden al exministro de Obras Públicas, Édgar Montaño", #Politico,
    "El Gobierno aprueba el Decreto 5516 que abroga el 5503 y mantiene la eliminación de la subvención",
    "Barcelona es el Supercampeón de España de la mano de Raphinha",
    "La diabetes no para de crecer en América Latina",
    "El agro perfila cuatro ejes para potenciar al sector durante 2026",
    "Gobierno reporta ahorro de $us 240 millones por eliminar subsidio a combustibles en 22 días",
    "Real Madrid pone fin al ciclo de Xabi Alonso tras perder la Supercopa",
    "Aún no hay ningún aprehendido por el asesinato de Aramayo y la Policía revisa cámaras de seguridad",
    "Santa Cruz bajo la lupa: investigan 35 denuncias de corrupción policial en diciembre",
    "Transition Design: cómo los diseñadores se convierten en agentes de cambio social, cultural y ecológico",
    "Primera precarnavalera cultural recorrerá las calles del centro cruceño",
    "La selección nacional perdio contra Japon",
    "La seleccion nacional gano su partido amistoso a Chile"
]
# seleccion - Seleccion - Selección - selección - SELECCION - SELECCIÓN (6 caracteristicas)
# seleccion - seleccion - selección - selección - seleccion - selección (2 caracteristicas)
#strip_accents='unicode'
#selección=seleccion (1 caracteristica)

#Eliminacion de palabras vacias (stopwords='spanish')



#N-GRAMAS
#UNIGRAMAS = AGRUPAMIENTO POR UNA CARACTERISTICA 
#La Selección nacional concentrará en Santa Cruz con vistas a los tres encuentros de preparación
#["Selección","nacional","concetrará", "Santa", "Cruz", "vistas", "tres", "encuentros", "preparación" ]
#Unigramas (n=1)
#seleccion
#nacional
#concetrará
#Bigramas (n=2)
#Seleccion nacional
#nacional concetrara
#Santa Cruz
#Trigrama (n=3)
#Seleccion nacional concetrará

#ngram_range=(1,2)

#Mejor equilibrio con datasets pequeños y medianos usar hasta bigramas
#dataset grande Trigramas - n-gramas

palabrasVacias=set(stopwords.words("spanish"))

def limpiezaDatos(texto):
    texto=texto.lower()
    #ELIMINACION DE NUMEROS SIMBOLOS Y CARCTERES
    texto=re.sub(r'[/()-*+''""]','',texto)

    tokenizacion=word_tokenize(texto,language="spanish")
    tokensLimpios=[
        palabra for palabra in tokenizacion if palabra not in palabrasVacias and len(palabra)>2
    ]
    return " ".join(tokensLimpios)

noticiasLimpias=[limpiezaDatos(noticia) for noticia in noticias]



categorias=[
    "Deportivo",
    "Cultural",
    "Politico",
    "Policial",
    "Policial",
    "Deportivo",
    "Economico",
    "Economico",
    "Politico",
    "Politico",
    "Politico",
    "Deportivo",
    "Cultural",
    "Economico",
    "Economico",
    "Deportivo",
    "Policial",
    "Policial",
    "Cultural",
    "Cultural",
    "Deportivo",
    "Deportivo"

]

#Convertir el texto en una representacion numerica
vectorizar=TfidfVectorizer(
    #lowercase=True,
    strip_accents='unicode',
    #stop_words=palabrasVacias,
    ngram_range=(1,2),
    max_features=5000,
    #min_df=2
)
entradasVectorizadas=vectorizar.fit_transform(noticiasLimpias)

#Separacion de los datos
#80% entrenamiento, 20% pruebas
xEntrenamiento,xPrueba,yEntrenamiento,yPrueba=train_test_split(entradasVectorizadas,categorias,test_size=0.30, random_state=42)

#Crear el modelo Perceptron multicapa
modelo=MLPClassifier(
    hidden_layer_sizes=(30,15),
    activation="relu",
    #OPTIMIZADOR Adam
    solver="adam",
    max_iter=500,
    alpha=0.0005,
    #early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    random_state=42

)

#Entrenar el modelo
modelo.fit(xEntrenamiento,yEntrenamiento)

#verificar la presicion del modelo
predicciones=modelo.predict(xPrueba)

precision=accuracy_score(yPrueba,predicciones)
print(f"Precision del modelo es: {precision*100}%")


#reporte completo de la clasificacion por cada clase
print("Reporte de Clasificacion")
print(classification_report(yPrueba,predicciones))

#Predicciones con nuevos valores
nuevasNoticias=[
    "Refuerzan seguridad en la cárcel de Palmasola para evitar violencia",
    "El 69 porciento de los consumidores realizan sus compras en cadenas de farmacias",
    "La Selección nacional concentrará en Santa Cruz con vistas a los tres encuentros de preparación",
    "Bolivia ante Brasil, Argentina, Perú y Ecuador en la Seleccion Nacional Sub-20 femenino",
    "Fiscalía pide cárcel para cuñado de Yuvinca y afirma que el crimen ocurrió dentro del entorno familiar Uno de los puntos más sensibles señalados por la Fiscalía es que el crimen no habría implicado un secuestro ni el traslado de la menor a otro lugar, sino que, según los indicios recolectados, ocurrió dentro de un entorno familiar",
]

#Vectorizar los datos nuevos
nuevasNoticiasVectorizadas=vectorizar.transform(nuevasNoticias)

prediccionesNuevas=modelo.predict(nuevasNoticiasVectorizadas)

print("Predicciones nuevas")
for noticia, categoria in zip(nuevasNoticias,prediccionesNuevas):
    print(f"Noticia: {noticia} - Categoria Predicha: {categoria}" )
