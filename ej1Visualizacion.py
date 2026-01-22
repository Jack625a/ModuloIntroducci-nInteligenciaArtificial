import matplotlib.pyplot as plt

#Estructura basica de un grafico
#1 datos (x,y)
#2 Etiquetas
#3 Titulo
#4. Activacion del grafico (show)
ventas=[150,120,110,200,205] #y
meses=[1,2,3,4,5] #x

#Graficos de lineas (plot)
#Grafico de dispersion (scatter)
#Grafico de barras (Modelo de Clasificacion) (bar)

plt.bar(meses,ventas)
plt.xlabel("Mes")
plt.ylabel("Ventas")
plt.title("Ventas por Mes")
plt.show()
