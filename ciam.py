import sys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

#Funcion para leer el archivo .dat
def leerArchivo(nombre):
    binaryFile = open(nombre, mode='rb')#Abrimos el archivo en modo binario
    data = np.fromfile(binaryFile, dtype='d') # reads the whole file
    binaryFile.close()
    return data
        
#Codigo para graficar los valores
def graficar(data):
    #data.ndim
    x = np.arange(0,len(data))#Generamos los valores de x
    def f(x):
        return np.take(data,x)#Obtenemos la correspondencia del vector
    plt.plot(x,f(x))#Graficamos
    plt.show()#Mostramos en pantalla

#Funcion para normalizar los valores
def normalizar(data):
    max=maxValor(data)#Se calcula el valor maximo del vector
    for p in range(len(data)):
        data[p]=(data[p]*256)/max  #Formula para normalizar los valores de [0, 255]

#Funcion para promediar las muestras del arreglo
def promediar(dataF):
    j=0
    dataP = np.empty(int((len(dataF))/4))#Vector que contendra los valores promedio
    for i in range(0,(len(dataF)),4):
        dataP[j]=(dataF[i]+dataF[i+1]+dataF[i+2]+dataF[i+3])/4#Promedio de los valores
        j+=1;
    return dataP

#Funcion que calcula el maximo valor de un vector    
def maxValor(data):
    max=0.0
    for i in range(len(data)):
        if(data[i]>max):
            max=data[i]
    return max

#Transformacion del arreglo unidimensional a bidimencional
def transformarR1R2(data,fil,col):
    #dim=int(math.sqrt(len(data)))
    #matrixB = np.empty((fil, col))
    #print("Dim de la matrixB: ",matrixB.shape)
    matrixB=np.reshape(data,(fil,col))#
    return matrixB

#Transformacion del arreglo bidimensional a tridimencional    
def transformarR2R3(matrixB):
    matrixT = np.empty((matrixB.shape[0], matrixB.shape[1],3))
    for m in range(3):
        for n in range(matrixB.shape[0]):
            for p in range(matrixB.shape[1]):
                matrixT[n][p][m]=matrixB[n][p]
    return matrixT

#Código para girar las filas impares
def girar(matrixB):
    turn=False
    aux = np.empty((matrixB.shape[1]))
    for i in range(matrixB.shape[0]):
        if(turn==True):
            for k in range(matrixB.shape[1]):
                aux[k]=matrixB[i][k]
            for l in range(matrixB.shape[1]):
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
    
def acoplar(numAcoplo,matrixBG):
    acoplo=True
    for i in range(matrixBG.shape[0]):
        if(acoplo==True):
            matrixBG[i,:]=fase(numAcoplo,matrixBG[i,:])
        acoplo=not(acoplo)
    return matrixBG

#Codigo para crear la imagen a partir de una matriz R3
def crearImagen(imagen):
    cv2.imwrite("MyImage.png",imagen)
    #plt.imshow(imagen)
    #plt.show()
    """cv2.imshow("Imagen",imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    
def crearImagenPil(data):
    a = []
    r = 0
    g = 255
    b = 0

    for i in range(600*512):
        colorTuple = (r, int(data[i]), b)
        a.append(colorTuple)

    newimage = Image.new('RGB', (600, 512))
    newimage.putdata(a)
    newimage.show()

def crearImagenPlot(data):
    max=maxValor(data)
    dataAux = np.empty(len(data))
    for p in range(len(data)):
        dataAux[p]=(data[p]*1)/max
    matrixB=transformarR1R2(dataAux,512,600)
    matrixT=transformarR2R3(matrixB)
    imagen=elegirCanal("G",matrixT)
    plt.imshow(imagen)
    plt.show()

    
#Codigo para recortar la imagen, quitamos elementos la matriz
def recortar(matrixCouple,numRecorte):
    columnaD=(matrixCouple.shape[1]-1)-(np.arange(0,numRecorte))
    columnaI=np.arange(0,numRecorte)
    matrixR=np.delete(matrixCouple, columnaD, axis=1)
    matrixR=np.delete(matrixR, columnaI, axis=1)
    return matrixR
    
#Funcion para elegir el canal de la matriz       
def elegirCanal(canal,matrixT):
    imagen = np.empty((matrixT.shape[0], matrixT.shape[1],3))
    if(canal=='R'):
        indices=[1,2]
    if(canal=='G'):
        indices=[0,2]
    if(canal=='B'):
        indices=[0,1]
    if(canal=='RGB'):
        indices=[]
    for m in indices:
        for n in range(matrixT.shape[0]):
            for p in range(matrixT.shape[1]):
                matrixT[n][p][m]=0
    imagen=matrixT
    return imagen
    
def construirImagen(data,canal,desp):
    crearImagenPlot(data)
    normalizar(data)
    crearImagenPil(data)
    #print("Data nueva: ",len(data))
    print("Maximo valor: ",maxValor(data))
    #graficar(data)
    print("data normalizado:",data)
    matrixB=transformarR1R2(data,512,600) #Covension R1 --> R2
    print("MatrixB: ",matrixB)
    matrixBG=girar(matrixB)
    print("MatrixBG: ",matrixBG)
    matrixCouple=acoplar(int(desp),matrixBG)
    matrixR=recortar(matrixCouple,int(44))
    print("La matriz recortada: ",matrixR.shape)
    matrixT=transformarR2R3(matrixR) #Covension R2 --> R3
    print("MatrixT: ",matrixT[:,:,2])
    imagen=elegirCanal(canal,matrixT)
    crearImagen(imagen)

def main(params):
    numParams=len(sys.argv)
    if(numParams==1):
        print("Archivo de lectura no especificado")
    if(numParams==2):
        data=leerArchivo(params[1])
        #dataP=promediar(data)
        construirImagen(data,"RGB",0)
    if(numParams==3):
        data=leerArchivo(params[1])
        #graficar(data)
        print("Maximo valor: ",maxValor(data))
        if(params[2]=="-g"):
            graficar(data)
        else:
            #dataP=promediar(data)
            construirImagen(data,params[2],0)
    if(numParams==4):
        data=leerArchivo(params[1])
        #dataP=promediar(data)
        construirImagen(data,params[2],params[3])
    if(numParams==5):
        data=leerArchivo(params[1])
        if(params[2]=="-p"):
            dataP=promediar(data)
            construirImagen(dataP,params[4],params[5])
                
main(sys.argv)