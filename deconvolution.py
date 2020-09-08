from time import time
from imageFunctions import mostrarImagen
from imageFunctions import normalizar
from imageFunctions import imgReadCv2
from imageFunctions import guardarImagen
from imageFunctions import elegirCanal
from deconvTF import deconvolveTF

to=time()

img = imgReadCv2('Images/310114 1hz-23_C003Z001.bmp',2)
psf = imgReadCv2('PSF/PSF_BW6.bmp',2)

deconv = deconvolveTF(img, psf,10)

#mostrarImagen("Convolve",img,False)
deconvN = normalizar(deconv)
deconvNR = elegirCanal("B",deconvN)
mostrarImagen("Deconvolucion",deconvNR,True)

#guardarImagen("Deconvolutions/Deconvolve_310114_1hzC3.bmp",outdeconv)
#guardarImagen("Deconvolve_310114_1hzC3.bmp",deconvN)

tf=time()
tt=tf-to
print("El tiempo total fue: ",tt/60, "minutos")
print("Ha terminado la ejecucion")
