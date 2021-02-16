import interfaceTools as it
import deconvolution as dv

entryIterations,entryWeight,dropdownImg, dropdownPSF = (None,None,None,None)

def deconvolutionEvent():
	global entryIterations, entryWeight, dropdownImg, dropdownPSF
	opcDeconv = it.NewWindow('Deconvolution Options','300x180') #Objeto de la clase NewWindow
	opcDeconv.createLabel('Image: ',20,20)
	opcDeconv.createLabel('PSF: ',20,50)
	opcDeconv.createLabel('Iterations: ',20,80)
	opcDeconv.createLabel('Weight: ',20,110)
	entryIterations = opcDeconv.createEntry('50',110,80)
	entryWeight = opcDeconv.createEntry('20',110,110)
	dropdownImg = opcDeconv.createCombobox(110,20)
	dropdownPSF = opcDeconv.createCombobox(110,50)
	opcDeconv.createButton('OK', deconvolutionKernel, 'bottom')

def deconvolutionKernel():
	global entryIterations, entryWeight, dropdownImg, dropdownPSF
	dv.deconvolutionMain(it.filesPath[dropdownImg.current()],it.filesPath[dropdownPSF.current()],entryIterations.get(),entryWeight.get())

#Se crea la ventana principal del programa
it.createWindowMain()
#Se crea menu desplegable
menu = it.createMenu()
#Se añaden las opciones del menu
opc1 = it.createOption(menu)
it.createCommand(opc1, "Abrir", it.openFile)
it.createCommand(opc1, "Guardar", it.saveFile)
it.createCommand(opc1, "Salir", it.mainWindow.quit)
it.createCascade(menu, 'Archivo', opc1)

opc2 = it.createOption(menu)
it.createCommand(opc2, "Deconvolution", deconvolutionEvent)
#it.createCommand(opc2, "Guardar", it.saveFile)
#it.createCommand(opc2, "Salir", mainWindow.quit)
it.createCascade(menu, 'Image', opc2)

statusBar = it.createStatusBar()
statusBar['text']=dv.message

it.mainWindow.mainloop()