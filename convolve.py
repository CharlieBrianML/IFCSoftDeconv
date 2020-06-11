import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
from time import time
import scipy.signal
import cv2

def lecturaX():
	binaryFile = open("imagen1.dat", mode='rb')#Abrimos el archivo en modo binario
	x = np.fromfile(binaryFile, dtype='d') # reads the whole file
	return x
	
def lecturaY():
	binaryFile = open("PSF.png", mode='rb')#Abrimos el archivo en modo binario
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
	
def transformarR1R2(data,fil,col):
    #dim=int(math.sqrt(len(data)))
    #matrixB = np.empty((fil, col))
    #print("Dim de la matrixB: ",matrixB.shape)
    matrixB=np.reshape(data,(fil,col))
    return matrixB
	
#Transformacion del arreglo unidimensional a bidimencional
def transformarR1R2(data,fil,col):
	matrixB=np.reshape(data,(fil,col))#
	return matrixB

def main():
	#obj=lecturaX()
	#psf=lecturaY()
	img=cv2.imread("Flor.jpg")
	psf=cv2.imread("PSF.png")
	#num=5.0
	#psf[0]=psf[0]*5.0
	#print(type(num))
	print('El tipo es: ',type(psf),type(psf[0]), np.shape(psf), len(psf.shape))
	print(len(psf))
	psf=np.reshape(psf[:,:,2],65536)
	img=np.reshape(img[:,:,2],65536)
	print('La nueva psf: ',len(psf.shape),' ', len(psf))
	conv=np.convolve(psf,img)
	print('len: ',len(conv))
	conv=transformarR1R2(conv[0:131044],362,362)
	deconv=scipy.signal.deconvolve( conv, psf )
	print('Deconvolve: ',deconv,'\n', np.shape(deconv[1]))
	
	#psf=float(psf)
	#print(type(psf))
	#psf=transformarR1R2(psf,19,13)
	#dim=int(math.sqrt(len(data)))
	cv2.imwrite("Convolve.png",deconv[1])
'''	print(psf)
	graficar(psf)
	obj=normalizar(obj)
	objF=transformadaF(obj)
	psfF=transformadaF(psf)
	#imagF=convolucionF(objF,psfF)
	imagF=objF*psfF
	print(imagF)
	for i in range(len(objF)):
		objF[i]=imagF[i]/psfF[i]
	obj=np.fft.ifft(objF)
	print("Este es obj: \n",obj.real)
	objR2=transformarR1R2(obj.real,256,256)
	cv2.imwrite("Convolve.png",psf)'''

main()