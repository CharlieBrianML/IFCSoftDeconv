from time import time
import imageFunctions as imf
from deconvTF import deconvolveTF

to=time()

img = imf.imgReadCv2('Images/310114 1hz-23_C003Z001.bmp',0) #Leemos la imagen a procesar 
psf = imf.imgReadCv2('PSF/PSF_BW8.bmp',0) #Leemos la psf de la imagen

deconv = deconvolveTF(img, psf,40) #Funcion de deconvolucion de imagenes

#mostrarImagen("Convolve",img,False)
deconvN = imf.normalizar(deconv) #Se normaliza la matriz 
deconvNR = imf.elegirCanal("R",deconvN) #Se eligue un canal RGB
imf.mostrarImagen("Deconvolucion",deconvNR,True)

imf.guardarImagen("Deconvolutions/Deconvolve_310114_1hz_Blue40Iterations.bmp",deconvNR) #Guardamos el resultado en memoria
#imf.guardarImagen("C:/Users/charl/Desktop/Deconvolve_310114_1hz_10Iterations.bmp",deconvN)

tf=time()
tt=tf-to
print("El tiempo total fue: ",tt/60, "minutos")
print("Ha terminado la ejecucion")
