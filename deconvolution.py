from time import time
import imageFunctions as imf
from deconvTF import deconvolveTF
from getpass import getuser
import tiff as tif
import os
import sys

color = ['g','r','b']

def deconvolutionTiff(stack,stackpsfs,iterations):
	deconv_list=[]
	for i in range(len(stack)):
	# for i in [1]:
		psf=imf.escalaGrises(stackpsfs[i])
		img=imf.escalaGrises(stack[i])
		for j in range(1):
			tv_denoised = imf.denoisingTV(img, 20)
			deconv = deconvolveTF(tv_denoised,psf,iterations) #Funcion de deconvolucion de imagenes
			#deconv = deconvolveTF(img,psf,iterations) #Funcion de deconvolucion de imagenes
			deconvN = imf.normalizar(deconv) #Se normaliza la matriz 
			deconvC = imf.elegirCanal(color[i],deconvN) #Se eligue un canal RGB
			#imf.guardarImagen('C:/Users/charl/Desktop/Deconvolve_'+str(i)+'310114_1hz.bmp',deconvC)
			print(deconvN.shape)
			#deconv8B=imf.rescaleSkimage(deconvN)
			deconv8B=deconvC.astype("uint16")
			deconv_list.append(deconv8B)
	return deconv_list
	
def deconvolutionRGB(img,psf,iterations):
	imgG=imf.escalaGrises(img)
	deconv=deconvolveTF(imgG, psf, iterations) #Funcion de deconvolucion de imagenes
	deconvN=imf.normalizar(deconv)
	deconvC=imf.elegirCanal('r',deconv)
	return deconvC
	
def deconvolution1Frame(img,psf,iterations):
	deconv=deconvolveTF(img, psf, iterations) #Funcion de deconvolucion de imagenes
	deconvN=imf.normalizar(deconv)
	return deconvN
	
to=time()

# Construct the argument parse and parse the arguments
imgpath, psfpath, i = sys.argv[1:4]
nameFile, extention = os.path.splitext(imgpath)
nameFile = nameFile.split('/')[len(nameFile.split('/'))-1]
dir_path = os.path.dirname(os.path.realpath(sys.argv[0])) 
path = "C:/Users/"+getuser()+"/Desktop"

if(extention=='.tif'):
	imgs = tif.leerTiff(imgpath)
	psfs = tif.leerTiff(psfpath)
	deconv=deconvolutionTiff(imgs,psfs,i)
	tif.imgtoTiff(tif.imgtoMatrix(deconv),os.path.join(path,'Deconvolution_'+nameFile+'.tif'))
	#tif.imgtoTiff(deconv,os.path.join(path,'Deconvolution_'+nameFile+'.tif'))
else:
	if(extention=='.jpg' or extention=='.png' or extention=='.bmp'):
		img = imf.imgReadCv2(imgpath) #Leemos la imagen a procesar 
		psf = imf.imgReadCv2(psfpath) #Leemos la psf de la imagen
		print(psf.shape)
		psf=imf.escalaGrises(psf)
		if(img.ndim>1):
			deconv=deconvolutionRGB(img,psf,i)
		else:
			deconv=deconvolution1Frame(img,psf,i)
		imf.guardarImagen(os.path.join(path,'Deconvolution_'+nameFile+'.bmp',deconv)
	else:
		print('La extension del archivo no es valida')

tf=time()
tt=tf-to
print("El tiempo total fue: ",tt/60, "minutos")
print("Ha terminado la ejecucion")
