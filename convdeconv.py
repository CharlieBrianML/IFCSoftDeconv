from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

def convolvek(image, kernel):

	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype='float32')

	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			roi = image[y - pad: y + pad + 1, x - pad: x + pad + 1]
			k = (roi * kernel).sum()
			output[y - pad, x - pad] = k

	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")
	#output = output.astype(int)

	return output
	
	
def deconvolveK(image, kernel):

	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype='float32')

	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			roi = image[y - pad: y + pad + 1, x - pad: x + pad + 1]
			k = (roi * kernel).sum()
			output[y - pad, x - pad] = k

	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")
	#output = output.astype(int)

	return output
	
def convolveF(img, psf):
	
	(iH, iW) = img.shape[:2]
	#convF = np.zeros((iH, iW), dtype='complex')
	
	imgF=np.fft.fft(img) #Transformada de Fourier de la imagen
	psfF=np.fft.fft(psf) #Transformada de Fourier de la psf
	#print("imgF: ",imgF[0][:])
	#print("psfF: ",psfF[0][:])
	convF=imgF*psfF #Convolucion en el espacio de Fourier
	print("convF: \n",convF,"\nLen: ", convF.shape)
	return convF	
	
def deconvolveF(convF, psf):
	
	(iH, iW) = convF.shape[:2] #Se obtiene las dimensiones de la matriz de convolucion
	imgF = np.zeros((iH, iW), dtype='complex')
	
	psfF=np.fft.fft(psf) #Se obtiene la transformada de Fourier
	for i in range(iH):
		for j in range(iW):
			imgF[i][j]=convF[i][j]/psfF[i][j] #Se hace la division punto a punto con la psf
			#if(np.abs(psf[i][j])==0.0):
			#	imgF[i][j]=0.0+0.0j
			#else:
			#	imgF[i][j]=convF[i][j]/psfF[i][j] #Se hace la division punto a punto con la psf
	print("imgF: \n",imgF)
	img=np.abs(np.fft.ifft(imgF)) #Se realiza la transformada invesa de Fourier
	print("img: \n",img,"\nLen: ", img.shape)
	return img

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# Construct average blurring kernels used to smooth the image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct the sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, 1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edge-like regions of an image
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobel1X = np.array((
    [-1, 0, -1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobel1Y = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")
	
# construct the Sobel y-axis kernel
miKernel = np.array((
    [2, 4, 1],
    [4, 3, 5],
    [1, 2, 3]), dtype="int")


def deconvolveFourier():
	image = cv2.imread(args["image"])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print("Imagen: \n",gray)

	psf = cv2.imread('PSF.png')
	psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
	print("PSF: \n",psf)

	miconv = convolveF(gray, psf)
	print("abs(miconv): \n",np.abs(miconv),"\nLen: ", (np.abs(miconv)).shape)
	outconv = rescale_intensity(np.abs(miconv), in_range=(0, 255))
	outconv = (outconv * 255).astype("uint8")

	mideconv = deconvolveF(miconv, psf)
	outdeconv = rescale_intensity(np.abs(mideconv), in_range=(0, 255))
	outdeconv = (outdeconv * 255).astype("uint8")

	cv2.imshow("original", gray)
	cv2.imshow("Convolucion", outconv)
	#cv2.imwrite("ImgConv.png",miconv)
	cv2.imshow("Deconvolucion", outdeconv)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
def deconvolveKernel():
	image = cv2.imread(args["image"])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print("Imagen: \n",gray)

	miconv = convolveK(gray, 0.05*miKernel)
	print("Convolucion: \n",miconv)

	#print("Determinante: ",np.linalg.det(miKernel))
	#miKernelInv=np.linalg.inv(miKernel)
	#print("Kernel Inverso: \n", miKernelInv)
	mideconv = deconvolveK(miconv, (np.linalg.inv(miKernel)))
	print("Deconvolucion: \n",mideconv)

	cv2.imshow("original", gray)
	cv2.imshow("Deconvolucion", miconv)
	cv2.imshow("Convolucion", 20*mideconv)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
deconvolveFourier()
#deconvolveKernel()