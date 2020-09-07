import numpy as np
import matplotlib.pyplot as plt
import cv2

def senoidal():
	n=1000
	tx=200
	w=2.0*np.pi/tx

	t=np.linspace(0,tx,n)
	s1=2.0*np.cos(2.0*w*t)
	s2=1.0*np.sin(30.0*w*t)
	s=s1+s2

	
def pulsoRec():
	t=np.linspace(-0.5,0.5,10000)
	u0 = lambda t: np.piecewise(t,t>=2,[1,0])
	u1 = lambda t: np.piecewise(t,t>=4,[1,0])
	u0 = u0(t-2)
	u1 = u1(t-4)
	u2 = u0-u1
	
def TF2(img): #Transformada de Fourier 2D
	return np.fft.fft2(img)
	
def TIF2(img): #Transformada inversa de Fourier 2D
	return np.ifft.fft2(img)
	
def freqEspectro(imgF,long): #Determina frecuencias de una imagen
	return np.fft.fftfreq(long, d=1.0)
	
def supFrecq2(imgF,freqs,val): #Atenua frecuencias de un espectro 2D
	img1D=np.reshape(imgF,imgF.shape[0]*imgF.shape[1])
	for i in (freqs):
		img1D[i]=val
	return np.reshape(img1D,(imgF.shape[0],imgF.shape[1]))
	
def graficarEspectro(freq,imgsF):
	for i in range(len(imgsF)):
		plt.figure(i)
		magnitude_spectrum_data = 20*np.log(np.reshape(np.abs(imgsF[i]),800*800))
		plt.vlines(freq,0,magnitude_spectrum_data)
		plt.title('Espectro Deconvolution')
	plt.show()
	
def main():
	img1 = cv2.imread('Deconvolutions/Deconvolve_verdeactro.png', 0)
	img2 = cv2.imread('Deconvolutions/Deconvolve_verdeactro2.png', 0)
	img3 = cv2.imread('Deconvolutions/Deconvolve_verdeactro3.png', 0)
	imgs=(img1,img2,img3)
	freq=freqEspectro(img1,img1.shape[0]*img1.shape[1])
	imgsF=TF2(imgs)
	graficarEspectro(freq,imgsF)

#main()