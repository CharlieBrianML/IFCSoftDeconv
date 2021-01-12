from tkinter import *    # Carga módulo tk (widgets estándar)
from tkinter import filedialog as fd
from tkinter import ttk  # Carga ttk (para widgets nuevos 8.5+)
from PIL import ImageTk, Image
import cv2 

def openFile():
	file = fd.askopenfilename(initialdir = '/', title = 'Seleccione archivo', filetypes = (('png files','*.png'),('all files','*.*')))
	
def saveFile():
	file = fd.asksaveasfilename(initialdir = '/',title = 'Seleccione archivo', defaultextension = '.png',filetypes = (('png files','*.png'),('all files','*.*')))
	#cv2.imwrite('C:/Users/charl/Desktop/miprueba.png',img)
	
def createWindowMain():
	# Define la ventana principal de la aplicación
	mainWindow = Tk() 
	mainWindow.geometry('500x50') # anchura x altura
	# Asigna un color de fondo a la ventana. 
	mainWindow.configure(bg = 'beige')
	# Asigna un título a la ventana
	mainWindow.title('IFCSoftDeconv')
	mainWindow.resizable(width=False,height=False)
	return mainWindow
	
def createMenu(mainWindow):
	#Barra superior
	menu = Menu(mainWindow)
	mainWindow.config(menu=menu)
	return menu
	
def createOption(menu):
	opc = Menu(menu, tearoff=0)
	return opc
	
def createCommand(opc, labelName, commandName):
	opc.add_command(label=labelName, command = commandName)
	return opc
	
def createCascade(menu, labelName, option):
	menu.add_cascade(label=labelName, menu=option)
	
def createButton(mainWindow, text, command, side):
	ttk.Button(mainWindow, text=text, command=command).pack(side=side)
	
def createEntry(mainWindow, stringVar):
	entry = ttk.Entry(mainWindow, state=DISABLED, textvariable=stringVar)
	return entry
	
def createStringVar():
	nombre = StringVar()
	return nombre
	
def createNewWindow(mainWindow):
	#Ventana de la imagen
	venImg = Toplevel(mainWindow)
	#venImg.geometry('500x500') # anchura x altura
	#venImg.configure(bg = 'beige')
	venImg.title('Image')
	#venImg.resizable(width=False,height=False)
	return venImg
	
def placeImage(venImg, img):
	panel2 = Label(venImg, image = img)
	panel2.pack()
	#panel2.pack(side = "bottom", fill = "both", expand = "yes")
	#venImg.mainloop()