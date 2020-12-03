from time import time
from progress.bar import Bar, ChargingBar
import imageFunctions as imf
from deconvTF import deconvolveTF
import tiff as tif
import os
import sys
import tifffile
import numpy as np

def deconvolutionTiff(img,psf,iterations,weight):
	#deconv_list=img
	
	if(img.ndim==3):
		for c in range(img.shape[0]):
			img_denoised = imf.denoisingTV(img[c,:,:], weight)
			deconv = deconvolveTF(img_denoised,psf[c,:,:],iterations) #Funcion de deconvolucion de imagenes
			deconvN = imf.normalizar(deconv) #Se normaliza la matriz 
			print('Channel ',c+1,' deconvolved')
			deconv_list[c,:,:]=deconvN
	if(img.ndim==4):
		if(imf.istiffRGB(img.shape)):
			for c in range(img.shape[0]):
				deconv= deconvolutionRGB(img[c,:,:,:],psf[c,:,:], iterations, weight) #Funcion de deconvolucion de imagenes
				print('Channel ',c+1,' deconvolved')
				deconv_list[c,:,:,:]=deconv
		else:
			deconv_list=np.zeros((img.shape[0],img.shape[3],img.shape[1],img.shape[2]), dtype="int16")
			for c in range(img.shape[3]):
				bar = Bar("\nChannel "+str(c+1)+' :', max=img.shape[0])
				for z in range(img.shape[0]):
					img_denoised = imf.denoisingTV(img[z,:,:,c], weight)
					deconv= deconvolveTF(img_denoised,psf[z,:,:,c], iterations) #Funcion de deconvolucion de imagenes
					deconvN = imf.normalizar(deconv) #Se normaliza la matriz 
					deconv_list[z,:,:,c]=deconvN
					bar.next()
				bar.finish()
	if(img.ndim==5):
		for c in range(img.shape[0]):
			for z in range(img.shape[1]):
				deconv= deconvolutionRGB(img[z,c,:,:,:],psf[z,c,:,:,:], iterations) #Funcion de deconvolucion de imagenes
				deconv_list[z,c,:,:,:]=deconv
			print('Channel ',c+1,' deconvolved')	
	return deconv_list
	
def deconvolutionRGB(img,psf,iterations,weight):
	imgG=imf.escalaGrises(img)
	img_denoised = imf.denoisingTV(imgG,weight)
	deconv=deconvolveTF(img_denoised, psf, iterations) #Funcion de deconvolucion de imagenes
	deconvN=imf.normalizar(deconv)
	deconvC=imf.elegirCanal('r',deconv)
	return deconvC
	
def deconvolution1Frame(img,psf,iterations):
	img_denoised = imf.denoisingTV(imgG,weight)
	deconv=deconvolveTF(img_denoised, psf, iterations) #Funcion de deconvolucion de imagenes
	deconvN=imf.normalizar(deconv)
	return deconvN
	
to=time()

# Construct the argument parse and parse the arguments
imgpath, psfpath, i , weight = sys.argv[1:5]
if (os.path.exists(imgpath) and os.path.exists(psfpath)):
	nameFile, extImage = os.path.splitext(imgpath) #Se separa el nombre de la imagen y la extesion
	extPSF = os.path.splitext(psfpath)[1] #Se obtiene la extesion de la psf
	nameFile = nameFile.split('/')[len(nameFile.split('/'))-1] #Extrae el nombre de la imagen si no se encuentra en el mismo direcctorio
	weight=int(weight)
	path = os.path.dirname(os.path.realpath(sys.argv[0])) #Direcctorio donde se almacenara el resultado
	#path = "C:/Users/"+os.getlogin()+"/Desktop"-
	savepath = os.path.join(path,'Deconvolutions/Deconvolution_'+nameFile+'.tif')
	
	if(extImage=='.tif'):
		tiff = tif.readTiff(imgpath)
		dimtiff = tiff.ndim
		psf = tif.readTiff(psfpath)
		tiffdeconv = tiff
		if(imf.validatePSF(tiff,psf)):
			print('\nFiles are supported\nStarting deconvolution')
			if(dimtiff==2):
				tiffdeconv = deconvolution1Frame(tiff,psf,i,weight)
			if(dimtiff==3):
				if(tif.istiffRGB(tiff.shape)):
					tiffdeconv = deconvolutionRGB(tiff,psf,i,weight)
				else:
					tiffdeconv = deconvolutionTiff(tiff,psf,i,weight)
			if(dimtiff==4):
				tiffdeconv = deconvolutionTiff(tiff,psf,i,weight)
		else:
			print('Wrong psf dimention, please enter a valid psf')
			exit()
		tifffile.imsave(savepath, tiffdeconv, imagej=True)
		print('Deconvolution successful, end of execution')
	else:
		if(extImage=='.jpg' or extImage=='.png' or extImage=='.bmp'):
			if(extPSF=='.jpg' or extPSF=='.png' or extPSF=='.bmp'):
				img = imf.imgReadCv2(imgpath) #Leemos la imagen a procesar 
				psf = imf.imgReadCv2(psfpath) #Leemos la psf de la imagen
				psf=imf.escalaGrises(psf)
				print('\nFiles are supported\nStarting deconvolution')
				bar = Bar("\nProcessing: "+nameFile+extImage, max=1)
				print('\n')
				if(img.ndim>1):
					warnings.filterwarnings('ignore', '.*',)
					deconv=deconvolutionRGB(img,psf,i,weight)
					bar.next()
					bar.finish()
				else:
					deconv=deconvolution1Frame(img,psf,i)
				imf.guardarImagen(os.path.join(path,'Deconvolution_'+nameFile+'.bmp'),deconv)
				#bar.finish()
				print('Deconvolution successful, end of execution')
		else:
			print('The file extension is not valid')
	tf=time()
	tt=tf-to
	print("Runtime: ",tt/60, "minutes")
else: 
	print('There is no file or directory of the image or psf')