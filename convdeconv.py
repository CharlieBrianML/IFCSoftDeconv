from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

def convolve(image, kernel):

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

	# return the output image
	return output
	
	
def deconvolve2(image, kernel):

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

	# return the output image
	return output
	
def convolveF(img, psf):
	
	(iH, iW) = img.shape[:2]
	(kH, kW) = psf.shape[:2]
	imgF = np.zeros((iH, iW), dtype='complex')
	
	imgF=np.fft.fft(img)
	psfF=np.fft.fft(psf)
	conv=imgF*psfF
	print("conv: \n",img,"\nLen: ", img.shape)
	return conv	
	
def deconvolveF(convF, psf):
	
	(iH, iW) = convF.shape[:2]
	(kH, kW) = psf.shape[:2]
	imgF = np.zeros((iH, iW), dtype='complex')
	
	#convF=np.fft.fft(conv)
	psfF=np.fft.fft(psf)
	#psfFP=psfF.sum()
	for i in range(iH):
		for j in range(iW):
			imgF[i][j]=convF[i][j]/psfF[i][j]
	print("imgF: \n",imgF)
	#img=np.abs(imgF)
	img=np.abs(np.fft.ifft(imgF))
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
	
miKernelInv = np.array((
    [1/25, 2/5, -17/25],
    [7/25, -1/5, 6/25],
    [-1/5, 0, 2/5]), dtype="float")

def prueba1():
	image = cv2.imread(args["image"])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print("Imagen: \n",gray)

	psf = cv2.imread('PSF.png')
	psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
	print("PSF: \n",psf)

	#opencvOutput = cv2.filter2D(gray, -1, smallBlur)
	#convolveOutput = convolve(gray, miKernel)
	#laplacianOutput = convolve(gray, laplacian)
	miconv = convolveF(gray, 0.00009*psf)
	print("Convolucion: \n",miconv)
	#output = rescale_intensity(miconv, in_range=(0, 255))
	#output = (output * 255).astype("uint8")

	#print("Determinante: ",np.linalg.det(miKernel))
	#miKernelInv=np.linalg.inv(miKernel)
	#print("Kernel Inverso: \n", miKernelInv)
	mideconv = deconvolveF(miconv, psf)
	#print("Deconvolucion: \n",mideconv)

	cv2.imshow("original", gray)
	#cv2.imshow("OpenCVFilter", opencvOutput)
	#cv2.imshow("smallBlur", convolveOutput)
	#cv2.imshow("laplacian", laplacianOutput)
	cv2.imshow("Convolucion", np.abs(miconv))
	#cv2.imwrite("ImgConv.png",miconv)
	cv2.imshow("Deconvolucion", 20*mideconv)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
def prueba2():
	image = cv2.imread(args["image"])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print("Imagen: \n",gray)

	#psf = cv2.imread('PSF2.png')
	#psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
	#print(psf)

	#opencvOutput = cv2.filter2D(gray, -1, smallBlur)
	#convolveOutput = convolve(gray, miKernel)
	#laplacianOutput = convolve(gray, laplacian)
	miconv = convolve(gray, 0.05*miKernel)
	print("Convolucion: \n",miconv)
	#output = rescale_intensity(miconv, in_range=(0, 255))
	#output = (output * 255).astype("uint8")

	#print("Determinante: ",np.linalg.det(miKernel))
	#miKernelInv=np.linalg.inv(miKernel)
	#print("Kernel Inverso: \n", miKernelInv)
	mideconv = deconvolve2(miconv, (np.linalg.inv(miKernel)))
	#print("Deconvolucion: \n",mideconv)

	cv2.imshow("original", gray)
	#cv2.imshow("OpenCVFilter", opencvOutput)
	#cv2.imshow("smallBlur", convolveOutput)
	#cv2.imshow("laplacian", laplacianOutput)
	cv2.imshow("Deconvolucion", miconv)
	cv2.imshow("Convolucion", 20*mideconv)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
prueba1()
#prueba2()