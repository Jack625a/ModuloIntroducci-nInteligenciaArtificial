import pandas as pd
import numpy as np

productosBase={
    "Televisor":8500,
    "Laptop":3200,
    "Tablet":800,
    "SmartWatch":500,
    "Auriculares":120
}

data=[]
for i in range(1000):
    prod=np.random.choice(list(productosBase.keys()))
    mes=np.random.randint(1,13)
    publicidad=np.random.randint(100,1000)
    precio=productosBase[prod]
    ventas=np.random.randint(5,260)
    total=precio*ventas
    ganancia=int(total-publicidad)

    data.append([prod,mes,publicidad,precio,ventas,total,ganancia])

dataFrame=pd.DataFrame(data,columns=["Producto","Mes","Publicidad","Precio","Ventas","Total","Ganancia"])
dataFrame.to_csv("datasetSimulado.csv", index=False)

