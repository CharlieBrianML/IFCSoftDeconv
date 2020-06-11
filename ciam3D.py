import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection="3d") #Dibujo de los ejes

#x(t)=sin(t)
#y(t)=cos(t)
#z(t)=t
z=np.linspace(0,10,100)
x=np.sin(z)
y=np.cos(z)
ax.plot3D(x,y,z,'red') #Helice circular 
ax.scatter3D(x,y,z)

'''def f(x,y):
	return 4-x**2-y**2
	
x=np.linspace(-5,5,40)
y=x
X,Y= np.meshgrid(x,y)
Z=f(X,Y)
#wireframe
ax.plot_wireframe(X,Y,Z) #Estructura metalica
#ax.plot_surface(X,Y,Z,cmap="viridis") #Color de la superficie'''


plt.show()