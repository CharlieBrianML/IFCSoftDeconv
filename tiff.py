import cv2
import tifffile
import numpy as np
from skimage import io

#Funcion que lee un archivo .tif
def leerTiff(fileTiff):
	img = io.imread(fileTiff) #Lee el archivo .tif
	frames = [] #Lista que contendrá los frames del .tif
	numFrames=img.shape[0] #Se determina el num. de frames que contiene el archivo
	for i in range(numFrames):
		frames.append(img[i,:,:,:])
	return frames #Se retorna una lista con los frames separados
	
#Funcion que convierte una matriz multidimensional a un archivo .tif
def imgtoTiff(imgs,savepath):
	tifffile.imsave(savepath,imgs) #[[[]]] -> .tif

#Funcion que convierte imagenes a matrices multidimensionales
def imgtoMatrix(img_list):
	cnt_num = 0
	for img in img_list: #Estraemos cada imagen de la lista
		#gray_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
		new_img = img[np.newaxis, ::] #Convertimos la imagen a una matriz multidimensional
		if cnt_num == 0:
			tiff_list = new_img
		else:
			tiff_list = np.append(tiff_list, new_img, axis=0) #Agregamos a la lista las imagenes convertidas en matrices
		cnt_num += 1
	return tiff_list

#imgs=leerTiff('Deconvolutions/Stack.tif')
#imgsM=imgtoMatrix(imgs)
#imgtoTiff(imgsM,'C:/Users/charl/Desktop/hola.tif')
