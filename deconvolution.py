from time import time
from time import sleep
import interfaceTools as it
from progress.bar import Bar, ChargingBar
import imageFunctions as imf
from deconvTF import deconvolveTF
import tiff as tif
import os
import sys
import tifffile
import numpy as np

message = ''

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
	

def deconvolutionMain(imgpath,psfpath,i,weight):
	global message
	to=time()
	# Construct the argument parse and parse the arguments
	#imgpath, psfpath, i , weight = sys.argv[1:5]
	if (os.path.exists(imgpath) and os.path.exists(psfpath)):
		nameFile, extImage = os.path.splitext(imgpath) #Se separa el nombre de la imagen y la extesion
		extPSF = os.path.splitext(psfpath)[1] #Se obtiene la extesion de la psf
		nameFile = nameFile.split('/')[len(nameFile.split('/'))-1] #Extrae el nombre de la imagen si no se encuentra en el mismo direcctorio
		weight=int(weight)
		path = os.path.dirname(os.path.realpath(sys.argv[0])) #Direcctorio donde se almacenara el resultado
		#path = "C:/Users/"+os.getlogin()+"/Desktop"-
		savepath = os.path.join(path,'Deconvolutions\Deconvolution_'+nameFile+'.tif')
		
		if(extImage=='.tif'):
			tiff = tif.readTiff(imgpath)
			dimtiff = tiff.ndim
			psf = tif.readTiff(psfpath)
			tiffdeconv = tiff
			if(imf.validatePSF(tiff,psf)):
				message = '\nFiles are supported\nStarting deconvolution'
				print(message)
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
				message = 'Wrong psf dimention, please enter a valid psf'
				print(message)
				exit()
			tifffile.imsave(savepath, tiffdeconv, imagej=True)
			message = 'Deconvolution successful, end of execution'
			print(message)
		else:
			if(extImage=='.jpg' or extImage=='.png' or extImage=='.bmp'):
				if(extPSF=='.jpg' or extPSF=='.png' or extPSF=='.bmp'):
					img = imf.imgReadCv2(imgpath) #Leemos la imagen a procesar 
					psf = imf.imgReadCv2(psfpath) #Leemos la psf de la imagen
					psf=imf.escalaGrises(psf)
					message = '\nFiles are supported\nStarting deconvolution'
					print(message)
					it.statusbar['text']=message
					sleep(1)
					message = "\nProcessing: "+nameFile+extImage
					it.statusbar['text']=message
					sleep(1)
					bar = Bar(message, max=1)
					print('\n')
					if(img.ndim>1):
						#warnings.filterwarnings('ignore', '.*',)
						deconv=deconvolutionRGB(img,psf,i,weight)
						bar.next()
						bar.finish()
					else:
						deconv=deconvolution1Frame(img,psf,i)
					#imf.guardarImagen(os.path.join(savepath,'\Deconvolution_'+nameFile+'.bmp'),deconv)
					imf.guardarImagen(os.getcwd()+'\Deconvolutions\Deconvolution_'+nameFile+'.bmp',deconv)
					#print(savepath,'\Deconvolution_'+nameFile+'.bmp')
					#bar.finish()
					message = 'Deconvolution successful, end of execution'
					print(message)
					it.statusbar['text']=message
					sleep(1)
			else:
				message = 'The file extension is not valid'
				print(message)
		tf=time()
		tt=tf-to
		print("Runtime: ",tt/60, "minutes")
		it.statusbar['text']="Runtime: "+str(tt/60)+"minutes"
		sleep(1)
	else: 
		message = 'There is no file or directory of the image or psf'
		print(message)
	message = ''