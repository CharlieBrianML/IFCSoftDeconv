import numpy as np
import cv2
import matplotlib.pyplot as plt

binaryFile = open("imagen1.dat", mode='rb')#Abrimos el archivo en modo binario
#byte = binaryFile.read()
data = np.fromfile(binaryFile, dtype='d') # reads the whole file

cont=0
turn=False
aux = np.arange(0,255)
matrix = np.arange(0,40)
#print(data)
imagen_blanca = np.ones((255,255), dtype = np.uint8)*255
imagen_negra = np.zeros((255,255))
print(imagen_blanca)

#Transformacion del arreglo unidimensional a bidimencional
def transformar(data):
    contRow=0
    matrixB = np.empty((256, 256))
    aux = np.empty((1, 255))
    matrixB=np.reshape(data,(-1,256))
    return matrixB

#Codigo para graficar los valores
def graficar(data):
    valor=0
    x = np.arange(0,65536)#Generamos los valores de x
    def f(x):
        #valor=100000000e300*np.take(data,x)#Obtenemos la correspondencia del vector
        #data[x]=valor
        #return data[x]
        return np.take(data,x)#Obtenemos la correspondencia del vector
    plt.plot(x,f(x))#Graficamos
    plt.show()#Mostramos en pantalla

#Código para girar las filas impares
def girar(data):
    contG=0
    turn=False
    for i in range(65536):
            if (contG==255):
                    if(turn==False):
                            turn=True
                            for j in range(10):
                                    aux[j]=data[i+1+j]
                    else:
                            turn=False
                            data[i]=aux[-(cont+1)]
                    contG=0
            else:
                    if(turn==True):
                            data[i]=aux[-(cont+1)]
                    contG=contG+1

def crearImagen(matrixB):
	#img = cv2.imread("matrizRecorte.png")
	#print(img)
    #plt.rcParams['image.cmap'] = 'Blues_r' #Escala para la imagen
    cv2.imwrite("MyImage.png",imagen_blanca)
    #img = cv2.imread("MyImage.png")
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(img, cmap='gray')
    #imagen_blanca = np.ones((255,255))*0.8
    plt.imshow(imagen_blanca, vmin=0,vmax=1,cmap=plt.cm.Blues)
    plt.show()
    #cv2.imshow("Gray Scale Image", gray_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()"""

    
#Codigo para recortar la imagen, quitamos elementos la matriz
def recortar():
    recorte=2
    contRow=0
    indices = np.arange(0,16)
    for k in range(40):
            if (contRow==9):
                    contRow=0
            else:
                    if (contRow<recorte):
                            indices=k
                    else:
                            if (contRow>10-1-recorte):
                                    indices=k
    #matrix=np.delete(matrix,0)
    print(indices)

def normalizar(data):
    for p in range(65535):
        data[p]=(data[p]*256)/4.6
        #data[p]=data[p]*0.1

#graficar(data)
normalizar(data)
#graficar(data)
#girar(data)
#graficar(data)
matrixB=transformar(data) #Covension R1 --> R2
#matrixG=girar(matrixB)
crearImagen(matrixB)
binaryFile.close()

"""https://www.unioviedo.es/compnum/laboratorios_py/Intro_imagen/introduccion_imagen.html
https://claudiovz.github.io/scipy-lecture-notes-ES/advanced/image_processing/index.html
https://facundoq.github.io/courses/aa2018/res/04_imagenes_numpy.html"""
