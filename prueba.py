#from PIL import Image
import numpy as np
data = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[13,14,15,16,17,18],[13,14,15,16,17,18]])
data2 = np.array([1,2,3,4,5,6,7,8,9])
#print(data2)

binaryFile = open("imagen4.dat", mode='rb')#Abrimos el archivo en modo binario
dataF = np.fromfile(binaryFile, dtype='d') # reads the whole file
print("Longitud: ",len(dataF))
#print("Dimension: ",data.shape[1])

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

def promediar(dataF):
    j=0
    dataP = np.empty(int((len(dataF))/4))
    for i in range(0,(len(dataF)),4):
        dataP[j]=dataF[i]+dataF[i+1]+dataF[i+2]+dataF[i+3]
        j+=1;
    return dataP

def girar(data):
	turn=False
	aux = np.empty((3))
	for i in range(3):
		if(turn==True):
			for k in range(3):
				aux[k]=data[i][k]
			for l in range(3):
				data[i][l]=aux[-(l+1)]
		turn=not(turn)
		
def right(dataDesf,numElm):
    dataAux = np.empty(numElm)
    for i in range(numElm):
        if(i==(numElm-1)):
            dataAux[i]=dataDesf[0]
        else:
            dataAux[i]=dataDesf[i+1]
    return dataAux
    
def left(dataDesf,numElm):
    dataAux = np.empty(numElm)
    for i in range(numElm):
        if(i==0):
            dataAux[0]=dataDesf[numElm-1]
        else:
            dataAux[i]=dataDesf[i-1]
    return dataAux
    
def fase(numFase,dataDesf):
    numElm=len(dataDesf)
    for j in range(numFase):
        dataDesf=left(dataDesf,numElm)
    return dataDesf
    
def acoplar(numAcoplo,data):
    acoplo=True
    for i in range(3):
        if(acoplo==True):
            data[i,:]=fase(numAcoplo,data[i,:])
        acoplo=not(acoplo)
    return data
    
def recortar(data,numRecorte):
    print(data)
    columna=5-(np.arange(0,numRecorte))
    print(columna)
    matrixR=np.delete(data, columna, axis=1)
    return matrixR
    
#dataCouple=acoplar(2,data)
#dataElim=recortar(data,2)
dataP=promediar(dataF)
print(len(dataP))