from skimage.exposure import rescale_intensity # if not installed -> pip3 install scikit-image

import numpy as np
import argparse
import cv2  # if not installed -> sudo apt install python3-opencv
import matplotlib.pyplot as plt

def convolve(image, kernel):
    # grab the spatial dimensions of the image, along with the spatial
    # dimension of the kernel
	print(np.shape(image))
	print(np.shape(kernel))
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	# allocate memory for the output image, taking care to "pad" the borders of
	# the input image so the spatial size (i.e., with and height) are not
	# reduced
	pad = (kW - 1) // 2
	print(pad)
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	print(image.shape)
	output = np.zeros((iH, iW), dtype='float32')
	print("Zeros: ",output.shape)

	# loop over the input image, "sliding" the kernel across each (x, y)-coordinate
	# from left-to-right and top to bottom
	#print(np.arange(pad, iH + pad))
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the "center" region of
			# the current (x, y)-coordinates dimensions
			roi = image[y - pad: y + pad + 1, x - pad: x + pad + 1]
			#print('roi: ',roi.shape)

			# perform thea actual convolution by taking the element-wise
			# multiplicative between the ROI and the kernel, then summing the
			# matrix
			k = (roi * kernel).sum()

			# stored the convolved value in the input (x, y)-coordinate of the
			# output image
			output[y - pad, x - pad] = k

	# rescale the output image to be in the range [0, 255]
	print(output)
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")
	print("En entero: \n",output)

	# return the output image
	return output

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

# load the input image and convert it to grayscale
#print("El argumento es: ",args["image"])
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

opencvOutput = cv2.filter2D(gray, -1, smallBlur)
convolveOutput = convolve(gray, smallBlur)
laplacianOutput = convolve(gray, laplacian)

cv2.imshow("original", gray)
cv2.imshow("OpenCVFilter", opencvOutput)
cv2.imshow("smallBlur", convolveOutput)
cv2.imshow("laplacian", laplacianOutput)

cv2.waitKey(0)
cv2.destroyAllWindows()