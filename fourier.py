import numpy as np
import matplotlib.pyplot as plt

def senoidal():
	n=1000
	tx=200
	w=2.0*np.pi/tx

	t=np.linspace(0,tx,n)
	s1=2.0*np.cos(2.0*w*t)
	s2=1.0*np.sin(30.0*w*t)
	s=s1+s2

	freq= np.fft.fftfreq(n)
	mascara=freq>0

	fft_calculo=np.fft.fft(s)
	fft_abs=2.0*np.abs(fft_calculo/n)

	plt.figure(1)
	plt.title('Signal original')
	plt.plot(t,s)

	plt.figure(2)
	plt.title('Signal fft')
	plt.plot(freq[mascara],fft_abs[mascara])
	plt.show()
	
def pulsoRec():
	t=np.linspace(-0.5,0.5,10000)
	u0 = lambda t: np.piecewise(t,t>=2,[1,0])
	u1 = lambda t: np.piecewise(t,t>=4,[1,0])
	u0 = u0(t-2)
	u1 = u1(t-4)
	u2 = u0-u1
	pulso=np.ones(10000,dtype=int)
	plt.figure(1)
	plt.plot(t,u2)
	
	freq= np.fft.fftfreq(10000)
	F=np.fft.fft(u0)
	#print(pulso)
	#print(F)
	IFI=np.abs(F)
	
	plt.figure(2)
	plt.plot(freq,IFI,'o')
	
	plt.show()

#pulsoRec()
senoidal()