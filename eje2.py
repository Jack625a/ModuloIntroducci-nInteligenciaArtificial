#API KEY GEMINI: 

#Paso1. Importacion de la libreria
import google.generativeai as genai

#Paso2. Configuracion de la api key
genai.configure(api_key="")

#Paso 3. Crear el Modelo
modelo=genai.GenerativeModel(model_name="gemini-3-flash-preview")

#Paso 4. Cargar el conocimiento (MEMORIA)
with open("simulacionData.txt","r", encoding="utf-8") as archivo:
    conocimientoLocal=archivo.read()

print("Bienvenido...")

while True:
    consulta=input("Usuario: ")

    if consulta.lower()=="salir":
        break

    contexto=f""" 
        Usa unicamente la siguiente informacion para responder:
        {conocimientoLocal}

        La pregunta del usuario:
        {consulta}
     """ #CONTEXTO
    
    #RAZONAMIENTO
    respuesta=modelo.generate_content(contexto)
    print("MODELO: ",respuesta.text)


    
