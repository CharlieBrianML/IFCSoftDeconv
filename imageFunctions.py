from skimage.exposure import rescale_intensity
import numpy as np
import cv2


def normalizar(data):
	max=np.amax(data)#Se calcula el valor maximo del vector
	for p in range(data.shape[0]):
		for m in range(data.shape[1]):
			data[p][m]=(data[p][m]*256)/max  #Formula para normalizar los valores de [0, 255]
	return data

#Implementa la funcion rescale_intensity de skimage
def rescaleSkimage(img):
	imgRescale = rescale_intensity(img, in_range=(0, 255))
	imgRescale = (imgRescale * 255).astype("uint8")
	return imgRescale

def mostrarImagen(nameWindow, img, close):
	cv2.imshow(nameWindow, img)
	if (close):
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
def guardarImagen(nameFile, img):
	cv2.imwrite(nameFile, img)
	
def imgReadCv2(nameImg,channel):
	return cv2.imread(nameImg,channel)
	
#def imgReadCv2(nameImg):
#	return cv2.imread(nameImg)

#Funcion para elegir el canal de la matriz       
def elegirCanal(canal,matrixT):
	img = np.zeros((matrixT.shape[0], matrixT.shape[1],3))
	if(canal=='R'):
		indices=[1,2]
		color=0
	if(canal=='G'):
		indices=[0,2]
		color=1
	if(canal=='B'):
		indices=[0,1]
		color=2
	if(canal=='RGB'):
		indices=[]
	if (matrixT.ndim==2):
		img[:,:,color]=matrixT
	else:
		for m in indices:
			matrixT[:,:,m]=0
		img=matrixT
	return img
	