
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import exposure
# from scipy import ndimage, signal
# from flowdec import data as fd_data
# from flowdec import restoration as fd_restoration
# from skimage.exposure import rescale_intensity
# import cv2

# def normalizar(data):
	# max=np.amax(data)#Se calcula el valor maximo del vector
	# for p in range(data.shape[0]):
		# for m in range(data.shape[1]):
			# data[p][m]=(data[p][m]*256)/max  #Formula para normalizar los valores de [0, 255]
	# return data

# # ===> Loading data
# img = cv2.imread('Deconvolutions/Deconvolve_310114_1hzC1.bmp')
# img2 = cv2.imread('Deconvolutions/Deconvolve_310114_1hzC2.bmp')
# img3 = cv2.imread('Deconvolutions/Deconvolve_310114_1hzC3.bmp')
# #cv2.imwrite("Deconvolutions/Deconvolve_verdeactro.jpg",img)

# img4=img
# print('Primero\n',img4)
# #img4[:,:,0]=img[:,:,1]
# img4[:,:,1]=img2[:,:,1]
# print('Segundo\n', img4)
# img4[:,:,2]=img3[:,:,2]
# print('Tercero\n',img4)
# #outimg4 = rescale_intensity(img4, in_range=(0, 255))
# #outimg4 = (outimg4 * 255).astype("uint8")
# #cv2.imwrite("Deconvolutions/Deconvolve_310114_1hzC4.bmp",img4)
# outimg4 = img4
# outimg4[:,:,0] = normalizar(img4[:,:,0])
# outimg4[:,:,1] = normalizar(img4[:,:,1])
# outimg4[:,:,2] = normalizar(img4[:,:,2])
# cv2.imshow("img4", outimg4)
# #cv2.imwrite("Deconvolutions/Deconvolve_310114_1hzNC4.bmp",img4)
# '''
# for i in range(1024):
	# for j in range(1024):
		# img4[i][j][1]=img2[i][j][1]
		
# for k in range(1024):
	# for l in range(1024):
		# img4[k][l][2]=img2[k][l][2]
# '''
# img[:,:,0]=0
# img[:,:,2]=0

# img2[:,:,0]=0
# img2[:,:,1]=0

# img3[:,:,1]=0
# img3[:,:,2]=0

# print(img4[:,:,0],'\n')
# print(img4[:,:,1],'\n')
# print(img4[:,:,2],'\n')
# print(img4)

# #cv2.imwrite("Deconvolutions/Deconvolve_310114_1hz.bmp",img)
# #outimg = rescale_intensity(img, in_range=(0, 255))
# #outimg = (outimg * 255).astype("uint8")
# outimg = img
# #cv2.imwrite("Deconvolutions/Deconvolve_310114_1hzC1.bmp",img)

# outimg[:,:,1] = normalizar(img[:,:,1])
# cv2.imshow("img", outimg)
# cv2.imwrite("Deconvolutions/Deconvolve_310114_1hzNC1.bmp",outimg)

# #cv2.imwrite("Deconvolutions/Deconvolve_310114_1hz2.bmp",img2)
# #outimg2 = rescale_intensity(img2, in_range=(0, 255))
# #outimg2 = (outimg2 * 255).astype("uint8")
# outimg2 = img2
# #cv2.imwrite("Deconvolutions/Deconvolve_310114_1hzC2.bmp",img2)

# outimg2[:,:,2] = normalizar(img2[:,:,2])
# cv2.imshow("img2", outimg2)
# cv2.imwrite("Deconvolutions/Deconvolve_310114_1hzNC2.bmp",outimg2)

# #cv2.imwrite("Deconvolutions/Deconvolve_310114_1hz3.bmp",img3)
# #outimg3 = rescale_intensity(img3, in_range=(0, 255))
# #outimg3 = (outimg3 * 255).astype("uint8")
# outimg3 = img3
# #cv2.imwrite("Deconvolutions/Deconvolve_310114_1hzC3.bmp",img3)

# outimg3[:,:,0] = normalizar(img3[:,:,0])
# cv2.imshow("img3", outimg3)
# cv2.imwrite("Deconvolutions/Deconvolve_310114_1hzNC3.bmp",outimg3)

# #cv2.imwrite("Deconvolutions/Deconvolve_310114_1hz4.bmp",img4)
# #outimg4 = rescale_intensity(img4, in_range=(0, 255))
# #outimg4 = (outimg4 * 255).astype("uint8")
# #cv2.imshow("img4", img4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
#Funcion para normalizar los valores
def normalizar(image):
#max=maxValor(data)#Se calcula el valor maximo del vector
	maxi=max(np.reshape(image,800*800))
	print('max: ', maxi)
	data=np.arange(800*800).reshape((800,800))
	for i in range(800):
		for j in range(800):
			data[i][j]=(image[i][j]*256)/maxi  #Formula para normalizar los valores de [0, 255]
	return data
'''
'''
resta= np.arange(800*800).reshape((800,800))
for i in range(800):
	for j in range(800):
		resta[i][j] = (float(img[i][j])) - (float(img2[i][j]))

#resta = img - img2
#print("img: ",img)
#print("img2: ",img2)
#print("resta: ",resta)
#plt.plot(np.arange(800*800),np.abs(np.reshape(resta,800*800)),'o')
plt.vlines(np.arange(800*800),0,np.abs(np.reshape(resta,800*800)))
plt.show()
#plt.imshow(data, cmap='gray')
#plt.show()
'''
'''
# ===> Deconvolution of data image
kernel = plt.imread('PSF/PSF_BW4.png')
print("La psf dim: ",kernel.shape)
#plt.plot(np.arange(800*800),np.reshape(kernel[:,:,1],800*800),'o')
#plt.show()
#plt.imshow(kernel, cmap='gray')
#plt.show()

iteration = 30
'''
'''
algo = fd_restoration.RichardsonLucyDeconvolver(data.ndim).initialize()
res = algo.run(fd_data.Acquisition(data=data, kernel=kernel), niter=iteration).data
#plt.imshow(res, cmap='gray')
#plt.show()
'''
'''
# ===> FFT of image
f = np.fft.fft2(img)
fdeconv = np.fft.fft2(imgdeconv[:,:,1])
k = np.fft.fft2(kernel[:,:,0])
freq=np.fft.fftfreq(800*800, d=1.0)

plt.figure(1)
#plt.plot(np.arange(800*800),np.reshape(np.abs(f),800*800),'o')
plt.vlines(freq,0,np.reshape(np.abs(f),800*800))
plt.title('Transformada Fourier ImagenConvolucion')
plt.figure(2)
#plt.plot(freq,np.reshape(np.abs(fdeconv),800*800),'o')
plt.vlines(freq,0,np.reshape(np.abs(fdeconv),800*800))
plt.title('Transformada Fourier ImagenDeconvolucion')
plt.figure(3)
#plt.plot(np.arange(800*800),np.reshape(np.abs(k),800*800),'o')
plt.vlines(freq,0,np.reshape(np.abs(k),800*800))
#plt.show()
'''
'''
for o in range(100):
	for i in range(800):
		for j in range(800):
			if(np.abs(f[i][j])==np.abs(k[i][j])):
				print("Hay una coincidencia")
			f[i][j]=f[i][j]-k[i][j]
'''		
'''	
for i in range(800):
	for j in range(800):
		if(np.abs(k[i][j])<0.1):
			if(np.abs(f[i][j])>0.1):
				f[i][j]=f[i][j]
			else:
				f[i][j]=0.0j
				print("Hay una coincidencia")
		else:
			f[i][j]=f[i][j]/500*k[i][j]

plt.figure(4)
plt.vlines(freq,0,np.reshape(np.abs(f),800*800))
plt.show()
'''
'''
print("La tranformada: ",f.shape)
fshift = np.fft.fftshift(f)
magnitude_spectrum_data = 20*np.log(np.abs(fshift))
# # ===> FFT of deconvoluted image
'''

#fI = np.fft.ifft2(img)
#fdeconvI = np.fft.ifft2(imgdeconv[:,:,1])

#imgR = rescale_intensity(img, in_range=(0, 255))
#imgR = (imgR * 255).astype("uint8")

#img2R = rescale_intensity(img2, in_range=(0, 255))
#img2R = (img2R * 255).astype("uint8")
'''
imgR=normalizar(img)
img2R=normalizar(img2)
print(img2R)
plt.imshow(imgR, cmap='gray')
plt.show()
'''
'''cv2.imshow("Convolucion", imgR)
cv2.imshow("Deconvolution", img2R)

cv2.waitKey(0)
cv2.destroyAllWindows()'''

'''
f = np.fft.fft2(res)
fshift = np.fft.fftshift(f)
magnitude_spectrum_res = 20*np.log(np.abs(fshift)+0.001)


data = rescale_intensity(data, in_range=(0, 255))
data = (data * 255).astype("uint8")
'''
#f = rescale_intensity(np.abs(f), in_range=(0, 255))
#f = (img * 255).astype("uint8")
'''
res = rescale_intensity(res, in_range=(0, 255))
res = (res * 255).astype("uint8")

magnitude_spectrum_data = rescale_intensity(magnitude_spectrum_data, in_range=(0, 255))
magnitude_spectrum_data = (magnitude_spectrum_data * 255).astype("uint8")
'''
'''
# ===> Showing the deconvolution
f1 = plt.figure(1)
plt.subplot(121), plt.imshow(data, cmap='gray')
plt.title('Original image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(res, cmap='gray')
plt.title('Deconvolution'), plt.xticks([]), plt.yticks([])
#plt.show()
'''
# # ===> Showing FFT original
#f2 = plt.figure(2)
#plt.subplot(121), plt.imshow(img, cmap='gray')
#plt.imshow(img, cmap='gray')
#plt.title('Original image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122), plt.imshow(np.abs(f), cmap='gray')
#plt.imshow(np.abs(f), cmap='gray')
#plt.title('FFT Original image'), plt.xticks([]), plt.yticks([])
#plt.show()
# # # ===> Showing FFT deconvolution
'''
cv2.imshow("Original", img)
cv2.imshow("Transformada", f)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
f3 = plt.figure(3)
plt.subplot(121), plt.imshow(res, cmap='gray')
plt.title('Original image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum_res, cmap='gray')
plt.title('FFT deconvolution'), plt.xticks([]), plt.yticks([])

plt.show()  # uncomment to show the figures
'''
'''
f1.savefig('./output_images/deconvolution.pdf', bbox_inches='tight')
f2.savefig('./output_images/originalFFT.pdf', bbox_inches='tight')
f3.savefig('./output_images/deconvolutionFFT.pdf', bbox_inches='tight')

plt.imsave('./output_images/original.png', data, cmap='gray')
plt.imsave('./output_images/deconvolution.png', res, cmap='gray')
plt.imsave('./output_images/original_FFT.png', magnitude_spectrum_data, cmap='gray')
plt.imsave('./output_images/deconvolution_FFT.png', magnitude_spectrum_res, cmap='gray')
'''




import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage import data, color, img_as_ubyte
#from skimage.filters import tv_denoise
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,denoise_wavelet, estimate_sigma)

img = cv2.imread('Images/ImgConv.png',0)

l = img_as_ubyte(color.rgb2gray(img))
l = l[230:290, 220:320]

cv2.imshow('Antes',img)
noisy = l + 0.4 * l.std() * np.random.random(l.shape)

tv_denoised = denoise_tv_chambolle(img, weight=0.5)

cv2.imshow('Despues',tv_denoised)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.figure(figsize=(8, 2))

plt.subplot(131)
plt.imshow(noisy, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.axis('off')
plt.title('noisy', fontsize=20)
plt.subplot(132)
#plt.imshow(tv_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.imshow(tv_denoised, cmap=plt.cm.gray)
plt.axis('off')
plt.title('TV denoising', fontsize=20)

tv_denoised = denoise_tv_chambolle(noisy, weight=50)
plt.subplot(133)
#plt.imshow(tv_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.imshow(tv_denoised, cmap=plt.cm.gray)
plt.axis('off')
plt.title('(more) TV denoising', fontsize=20)

plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0,right=1)
plt.show()