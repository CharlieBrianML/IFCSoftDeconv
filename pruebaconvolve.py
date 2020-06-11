import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
from time import time
import cv2

def lecturaX():
	binaryFile = open("imagen1.dat", mode='rb')#Abrimos el archivo en modo binario
	x = np.fromfile(binaryFile, dtype='d') # reads the whole file
	return x
	
def lecturaY():
	binaryFile = open("imagen2.dat", mode='rb')#Abrimos el archivo en modo binario
	y = np.fromfile(binaryFile, dtype='d') # reads the whole file
	binaryFile.close()
	return y

def convolucion(x,y):
	r= np.convolve(x,y) #Operacion de convolucion de numpy
	return r
	
def convolucionF(xF,yF):
	#xF=transformadaF(x)
	#yF=transformadaF(y)
	maxL=maxLong(xF,yF) #Vector de mayor correspondencia
	minL=minLong(xF,yF) #Vector de menor correspondencia
	x = np.arange(0,maxL) #Vector de recorrido
	for i in range(maxL):
		if(i<minL):
			x[i] = xF[i]*yF[i]
		else:
			x[i] = yF[i]*0
	return x
	
def maxLong(x,y):
	if(len(x)>len(y)):
		max=len(x)
	else:
		max=len(y)
	return max
	
def minLong(x,y):
	if(len(x)>len(y)):
		min=len(y)
	else:
		min=len(x)
	return min	
	
#Codigo para graficar los valores
def graficar(r):
	#data.ndim
	x = np.arange(0,len(r))#Generamos los valores de x
	def f(x):
		return np.take(r,x)#Obtenemos la correspondencia del vector
	plt.plot(x,f(x),'k-o')#Graficamos
	plt.show()#Mostramos en pantalla

#Funcion para normalizar los valores
def normalizar(data):
	max=maxValor(data)#Se calcula el valor maximo del vector
	for p in range(len(data)):
		data[p]=(data[p]*1)/max  #Formula para normalizar los valores de [0, 1]
	return data

#Funcion que calcula el maximo valor de un vector    
def maxValor(data):
    max=0.0
    for i in range(len(data)):
        if(data[i]>max):
            max=data[i]
    return max
	
#Codigo para aplicar la trasnformada de Fourier discreta
def transformadaF(data):
    vecF=np.fft.fft(data) #Operacion de transformada de Fourier de numpy
    return vecF

#Codigo para medir el tiempo de ejecucion de un segmento de codigo
def timeCode(time):
	if(time==True):
		to=time() #Definimos el tiempo inicial
	else:
		tf=time()
		tt=tf-to
	return tt

'''#Funcion para promediar las muestras del arreglo
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
    """cv2.imshow("Imagen",imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    
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
    
def construirImagen(data,canal,desp,fil,col):
    #crearImagenPlot(data)
    normalizar(data)
    #crearImagenPil(data)
    #print("Data nueva: ",len(data))
    #print("Maximo valor: ",maxValor(data))
    #graficar(data)
    #print("data normalizado:",data)
    matrixB=transformarR1R2(data,fil,col) #Covension R1 --> R2
    print("MatrixB: ",matrixB)
    matrixBG=girar(matrixB)
    print("MatrixBG: ",matrixBG)
    matrixCouple=acoplar(int(desp),matrixBG)
    matrixR=recortar(matrixCouple,int(44))
    print("La matriz recortada: ",matrixR.shape)
    matrixT=transformarR2R3(matrixR) #Covension R2 --> R3
    print("MatrixT: ",matrixT[:,:,2])
    imagen=elegirCanal(canal,matrixT)
    crearImagen(imagen)'''

def main():
	fw=np.array([1+4j,2+3j,3+3j,4+4j])
	#pulsoRec=np.ones(10)
	#x=np.linspace(0,1,2)
	#graficar(pulsoRec)
	#plt.plot(x,pulsoRec)
	#plt.show()
	#print("antes")
	#fw=transformadaF(pulsoRec)
	print("despue")
	#graficar(fw)
	print(fw)
	plt.plot(fw.real,fw.imag)
	plt.xlabel('Eje R')
	plt.ylabel('Eje I')
	plt.title('Plano de Argand')
	plt.grid()
	#set_ylim(-5, 5)
	#plt.show()
	print("antes")
	obj=lecturaX()
	psf=obj
	#x=normalizar(x)
	#graficar(x)
	#y=lecturaY()
	#y=normalizar(y)
	#graficar(y)
	#img=convolucion(obj,psf)
	#dominio=len(r)
	#print("El dominio de r: ",dominio)
	#graficar(r)
	#print("Tiempo de ejecucion de Convolucion: ", tt, " seg")
	objF=transformadaF(obj)
	psfF=transformadaF(psf)
	#imagF=convolucionF(objF,psfF)
	imagF=objF*psfF
	print(imagF)
	for i in range(len(objF)):
		objF[i]=imagF[i]/psfF[i]
	obj=np.fft.ifft(objF)
	print("Este es obj: \n",obj)
	cv2.imwrite("Convolve.png",obj.real)
	#print("Tiempo de ejecucion de Concolucion en el dominio de la frecuencia: ", t1+t2, " seg")
	#graficar(rF)
	
	
	
'''	xP = np.array([1])
	yP = np.array([1])
	t = np.array([0])
	for j in range(10600):
		print("Convolucion: ",j)
		xP=np.append(xP, 1)
		yP=np.append(yP, 1)
		to=time()
		xF=transformadaF(xP)
		yF=transformadaF(yP)
		cP=convolucionF(xF,yF)
		tf=time()
		tt=tf-to
		t=np.append(t, tt)
	graficar(t)
'''
main()