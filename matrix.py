import numpy as np
import cv2
import matplotlib.pyplot as plt

binaryFile = open("imagen1.dat", mode='rb')#Abrimos el archivo en modo binario
#byte = binaryFile.read()
data = np.fromfile(binaryFile, dtype='>d') # reads the whole file

cont=0
turn=False
#aux = np.arange(0,10)
matrix = np.arange(0,40)

#Transformacion del arreglo unidimensional a bidimencional
def transformar(data):
    contRow=0
    matrixB = np.empty((1, 255))
    #aux = np.array([[None, None, None, None, None, None, None, None, None, None]])
    aux = np.empty((1, 255))
    #matrixB = np.arange(0,10)
    for l in range(65025):
            if (contRow==254):
                aux[0][contRow]=data[l]*10e100
                #matrixB = numpy.array([[1, 2, 3], [4, 5, 6]])
                matrixB = np.append(matrixB, aux, axis = 0)
                contRow=0
            else:
                aux[0][contRow]=data[l]*10e100
                contRow=contRow+1
    #print(matrixB)
    #print(aux)
    return matrixB

#Codigo para graficar los valores
def graficar(data):
    valor=0
    x = np.arange(0,65536)#Generamos los valores de x
    def f(x):
        valor=100000000e300*np.take(data,x)#Obtenemos la correspondencia del vector
        data[x]=valor*1000000000
        return data[x]
        #return 100000000e300*np.take(data,x)#Obtenemos la correspondencia del vector
    plt.plot(x,f(x))#Graficamos
    plt.show()#Mostramos en pantalla

#Código para girar las filas impares
def girar():
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

def crearImagen(matrixB):
    #cv2.imwrite("MyImage.png",matrixB)
    #img = cv2.imread("MyImage.png")
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(img, cmap='gray')
    plt.imshow(matrixB, vmin=0,vmax=10)
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

graficar(data)
matrixB=transformar(data)
crearImagen(matrixB)
print (data)
binaryFile.close()

#Para mas informacion consulta: https://github.com/CharlieBrianML/InstitutoFisiologiaCelularSS
