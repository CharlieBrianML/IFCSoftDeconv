import numpy as np
import matplotlib.pyplot as plt

binaryFile = open("imagen1.dat", mode='rb')#Abrimos el archivo en modo binario
data = np.fromfile(binaryFile, dtype='>d') # reads the whole file

cont=0
turn=False
aux = np.arange(0,10)
matrix = np.arange(0,40)

#Codigo para graficar los valores
"""x = np.arange(0,65536)#Generamos los valores de x
def f(x):
    return 10000000e300*np.take(data,x)#Obtenemos la correspondencia del vector
plt.plot(x,f(x))#Graficamos
plt.show()#Mostramos en pantalla"""

#Código para girar las filas impares
for i in range(40):
	if (cont==9):
		if(turn==False):
			turn=True
			for j in range(10):
				aux[j]=matrix[i+1+j]
		else:
			turn=False
			matrix[i]=aux[-(cont+1)]
		cont=0
	else:
		if(turn==True):
			matrix[i]=aux[-(cont+1)]
		cont=cont+1
		
print (matrix)
binaryFile.close()
