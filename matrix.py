import numpy as np
import matplotlib.pyplot as plt
print("Lo que contiene el archivo es: ")
binaryFile = open("imagen1.dat", mode='rb')#Abrimos el archivo en modo binario
data = np.fromfile(binaryFile, dtype='>d') # reads the whole file
print (data)
x = np.arange(0,65536)#Generamos los valores de x
def f(x):
    return 10000000e300*np.take(data,x)#Obtenemos la correspondencia del vector
plt.plot(x,f(x))#Graficamos
plt.show()#Mostramos en pantalla
binaryFile.close()
