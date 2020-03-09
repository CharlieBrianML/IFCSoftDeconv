import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

binaryFile = open("imagen1.dat", mode='rb')#Abrimos el archivo en modo binario
data = np.fromfile(binaryFile, dtype='d') # reads the whole file

        
#Codigo para graficar los valores
def graficar(data):
    #data.ndim
    x = np.arange(0,65536)#Generamos los valores de x
    def f(x):
        return np.take(data,x)#Obtenemos la correspondencia del vector
    plt.plot(x,f(x))#Graficamos
    plt.show()#Mostramos en pantalla

#Funcion para normalizar los valores
def normalizar(data):
    for p in range(65535):
        data[p]=(data[p]*256)/4.6  #Formula para normalizar los valores de [0, 255]


#Transformacion del arreglo unidimensional a bidimencional
def transformarR1R2(data):
    #contRow=0
    matrixB = np.empty((256, 256))
    #aux = np.empty((1, 255))
    matrixB=np.reshape(data,(-1,256))
    return matrixB

#Transformacion del arreglo bidimensional a tridimencional    
def transformarR2R3(matrixB):
    matrixT = np.empty((256, 256,3))
    for m in range(3):
        for n in range(256):
            for p in range(256):
                matrixT[n][p][m]=matrixB[n][p]
    return matrixT

#Código para girar las filas impares
def girar(matrixB):
	turn=False
	aux = np.empty((256))
	for i in range(256):
		if(turn==True):
			for k in range(256):
				aux[k]=matrixB[i][k]
			for l in range(256):
				matrixB[i][l]=aux[-(l+1)]
		turn=not(turn)
	return matrixB
    
#Código para ajutar la fase de las filas 
def right(dataDesf,numElm):
    dataAux = np.empty(numElm)
    for i in range(numElm):
        if(i==(numElm-1)):
            dataAux[i]=dataDesf[0]
        else:
            dataAux[i]=dataDesf[i+1]
    return dataAux
    
def fase(numFase,dataDesf):
    numElm=len(dataDesf)
    for j in range(numFase):
        dataDesf=right(dataDesf,numElm)
    return dataDesf
    
def acoplar(numAcoplo,data):
    acoplo=True
    for i in range(256):
        if(acoplo==True):
            data[i,:]=fase(numAcoplo,data[i,:])
        acoplo=not(acoplo)
    return data

#Codigo para crear la imagen a partir de una matriz R3
def crearImagen(imagen):
    cv2.imwrite("MyImage.png",imagen)
    #plt.imshow(imagen)
    #plt.show()
    """cv2.imshow("Imagen",imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    
#Codigo para recortar la imagen, quitamos elementos la matriz
def recortar(matrixCouple,numRecorte):
    columna=255-(np.arange(0,numRecorte))
    matrixR=np.delete(matrixCouple, columna, axis=1)
    return matrixR
    
#Funcion para elegir el canal de la matriz       
def elegirCanal(canal,matrixT):
    imagen = np.empty((256, 256,3))
    if(canal=='R'):
        indices=[1,2]
    if(canal=='G'):
        indices=[0,2]
    if(canal=='B'):
        indices=[0,1]
    if(canal=='RGB'):
        indices=[]
    for m in indices:
        for n in range(256):
            for p in range(256):
                matrixT[n][p][m]=0
    imagen=matrixT
    return imagen


def main(params):
    numParams=len(sys.argv)
    for i in range(numParams):
        action=params[i]
        if(action=='-g'):
            graficar(data)
        """if(action=='R' or action=='G' or action=='B' or action=='RGB'):
            normalizar(data)
            matrixB=transformarR1R2(data) #Covension R1 --> R2
            matrixBG=girar(matrixB)
            matrixCouple=acoplar(20,matrixBG)
            matrixT=transformarR2R3(matrixCouple) #Covension R2 --> R3
            imagen=elegirCanal(action,matrixT)
            crearImagen(imagen)"""
        if(i==1):
            if(action=='R' or action=='G' or action=='B' or action=='RGB'):
                normalizar(data)
                matrixB=transformarR1R2(data) #Covension R1 --> R2
                matrixBG=girar(matrixB)
                matrixCouple=acoplar(int(params[i+1]),matrixBG)
                matrixR=recortar(matrixCouple,int(params[i+1]))
                matrixT=transformarR2R3(matrixCouple) #Covension R2 --> R3
                imagen=elegirCanal(action,matrixT)
                crearImagen(imagen)

main(sys.argv)

binaryFile.close()