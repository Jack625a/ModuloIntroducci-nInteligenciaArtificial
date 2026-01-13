from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


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
    "Primera precarnavalera cultural recorrerá las calles del centro cruceño"

]

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
    "Cultural"

]

#Convertir el texto en una representacion numerica
vectorizar=TfidfVectorizer()
entradasVectorizadas=vectorizar.fit_transform(noticias)

#Separacion de los datos
#80% entrenamiento, 20% pruebas
xEntrenamiento,xPrueba,yEntrenamiento,yPrueba=train_test_split(entradasVectorizadas,categorias,test_size=0.30)

#Crear el modelo Perceptron multicapa
modelo=MLPClassifier(
    hidden_layer_sizes=(10,),
    activation="relu",
    max_iter=1000
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