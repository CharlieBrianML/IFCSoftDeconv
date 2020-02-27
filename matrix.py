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
print(data)

#Transformacion del arreglo unidimensional a bidimencional
def transformar(data):
    contRow=0
    matrixB = np.empty((256, 256))
    #aux = np.array([[None, None, None, None, None, None, None, None, None, None]])
    aux = np.empty((1, 255))
    #matrixB = np.arange(0,10)
    matrixB=np.reshape(data,(-1,256))
    """for l in range(65536):
            if (contRow==255):
                aux[0][contRow]=data[l]*10e100
                #matrixB = numpy.array([[1, 2, 3], [4, 5, 6]])
                matrixB = np.append(matrixB, aux, axis = 0)
                contRow=0
            else:
                aux[0][contRow]=data[l]*10e100
                contRow=contRow+1
    #print(matrixB)
    #print(aux)"""
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
    plt.rcParams['image.cmap'] = 'Blues' #Escala para la imagen
    #cv2.imwrite("MyImage.png",matrixB)
    #img = cv2.imread("MyImage.png")
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(img, cmap='gray')
    #imagen_blanca = np.ones((255,255))*0.8
    plt.imshow(matrixB, vmin=0,vmax=1)
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

def normalizar():
    for m in range(65536):
        data[m]=data[m]*0.1

graficar(data)
girar(data)
normalizar()
print("\ndata:",data.size,data,"\n")
matrixB=transformar(data) #Covension R1 --> R2
#matrixG=girar(matrixB)
crearImagen(matrixB)
print ("\n",matrixB[0][2],"\n")
print (data[2])
binaryFile.close()

#Para mas informacion consulta: https://github.com/CharlieBrianML/InstitutoFisiologiaCelularSS
