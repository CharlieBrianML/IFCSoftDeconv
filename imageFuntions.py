from skimage.exposure import rescale_intensity


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
