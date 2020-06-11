import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import cv2

def deconv1():
	# let the signal be box-like
	signal = np.repeat([0., 1., 0.], 100)
	img=cv2.imread("Flor.jpg")
	psf=cv2.imread("PSF.png")
	psf=np.reshape(psf[:,:,2],65536)
	img=np.reshape(img[:,:,2],65536)
	# and use a gaussian filter
	# the filter should be shorter than the signal
	# the filter should be such that it's much bigger then zero everywhere
	gauss = np.exp(-( (np.linspace(0,50)-25.)/float(12))**2 )
	print (gauss.min())  # = 0.013 >> 0

	# calculate the convolution (np.convolve and scipy.signal.convolve identical)
	# the keywordargument mode="same" ensures that the convolution spans the same
	#   shape as the input array.
	#filtered = scipy.signal.convolve(signal, gauss, mode='same') 
	#filtered = np.convolve(signal, gauss, mode='same') 
	filtered = np.convolve(img, psf, mode='same') 

	#deconv,  _ = scipy.signal.deconvolve( filtered, gauss )
	deconv,  _ = scipy.signal.deconvolve( filtered, psf )
	#the deconvolution has n = len(signal) - len(gauss) + 1 points
	#n = len(signal)-len(gauss)+1
	n = len(signal)-len(gauss)+1
	# so we need to expand it by 
	s = int((len(signal)-n)/2)
	print(s)
	#on both sides.
	deconv_res = np.zeros(len(signal))
	deconv_res[s:len(signal)-s-1] = deconv
	deconv = deconv_res
	# now deconv contains the deconvolution 
	# expanded to the original shape (filled with zeros) 


	#### Plot #### 
	fig , ax = plt.subplots(nrows=4, figsize=(6,7))

	ax[0].plot(signal,            color="#907700", label="original",     lw=3 ) 
	ax[1].plot(gauss,          color="#68934e", label="gauss filter", lw=3 )
	# we need to divide by the sum of the filter window to get the convolution normalized to 1
	ax[2].plot(filtered/np.sum(gauss), color="#325cab", label="convoluted" ,  lw=3 )
	ax[3].plot(deconv,         color="#ab4232", label="deconvoluted", lw=3 ) 

	for i in range(len(ax)):
		ax[i].set_xlim([0, len(signal)])
		ax[i].set_ylim([-0.07, 1.2])
		ax[i].legend(loc=1, fontsize=11)
		if i != len(ax)-1 :
			ax[i].set_xticklabels([])

	plt.savefig(__file__ + ".png")
	plt.show()  

	
def deconv2():
	x = np.arange(0., 20.01, 0.01)
	y = np.zeros(len(x))
	y[900:1100] = 1.
	y += 0.01 * np.random.randn(len(y))
	c = np.exp(-(np.arange(len(y))) / 30.)

	yc = scipy.signal.convolve(y, c, mode='full') / c.sum()
	ydc, remainder = scipy.signal.deconvolve(yc, c)
	ydc *= c.sum()

	fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(4, 4))
	ax[0][0].plot(x, y, label="original y", lw=3)
	ax[0][1].plot(x, c, label="c", lw=3)
	ax[1][0].plot(x[0:2000], yc[0:2000], label="yc", lw=3)
	ax[1][1].plot(x, ydc, label="recovered y", lw=3)

	plt.show()
	
def deconv3():
	img=cv2.imread("Flor.jpg")
	psf=cv2.imread("PSF.png")
	psf=np.reshape(psf[:,:,2],65536)
	img=np.reshape(img[:,:,2],65536)
	# and use a gaussian filter
	# the filter should be shorter than the signal
	# the filter should be such that it's much bigger then zero everywhere
	gauss = np.exp(-( (np.linspace(0,50)-25.)/float(12))**2 )
	print (gauss.min())  # = 0.013 >> 0

	# calculate the convolution (np.convolve and scipy.signal.convolve identical)
	# the keywordargument mode="same" ensures that the convolution spans the same
	#   shape as the input array.
	#filtered = scipy.signal.convolve(signal, gauss, mode='same') 
	#filtered = np.convolve(signal, gauss, mode='same') 
	filtered = np.convolve(img, psf, mode='same') 

	#deconv,  _ = scipy.signal.deconvolve( filtered, gauss )
	deconv= scipy.signal.deconvolve( filtered, psf )
	#the deconvolution has n = len(signal) - len(gauss) + 1 points
	#n = len(signal)-len(gauss)+1
	n = len(img)-len(psf)+1
	# so we need to expand it by 
	s = int((len(img)-n)/2)
	print(s)
	#on both sides.
	deconv_res = np.zeros(len(img))
	deconv_res[s:len(img)-s-1] = deconv
	deconv = deconv_res
	# now deconv contains the deconvolution 
	# expanded to the original shape (filled with zeros)   


#deconv2()
#deconv1()
deconv3()