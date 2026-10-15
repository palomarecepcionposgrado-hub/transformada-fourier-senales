"""Actividad formativa 2: Fourier de señales muestreadas. Python 3, NumPy, Matplotlib.
Ejecutar: python simulacion_fourier.py
Las gráficas se guardan en el directorio 'graficas'.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'graficas'; OUT.mkdir(exist_ok=True)
fs=1024; T=2.0; N=int(fs*T); t=np.arange(N)/fs-T/2; dt=1/fs
freq=np.fft.fftshift(np.fft.fftfreq(N,d=dt))

def ft(x):
    # Aproximación de la transformada continua de la señal observada en ventana finita
    return dt*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))

def plot_signal(x,name,filename):
    X=ft(x); mag=np.abs(X); phase=np.angle(X)
    # La fase de bins con magnitud despreciable no es interpretable
    phase[mag < mag.max()*0.01]=np.nan
    fig,axes=plt.subplots(3,1,figsize=(10,8),layout='constrained')
    axes[0].plot(t,x,lw=1.6); axes[0].set(xlabel='Tiempo (s)',ylabel='Amplitud',title='Dominio del tiempo')
    axes[1].plot(freq,mag,lw=1.2); axes[1].set(xlabel='Frecuencia (Hz)',ylabel='|X(f)|',title='Magnitud espectral')
    axes[2].plot(freq,phase,'.',ms=3); axes[2].set(xlabel='Frecuencia (Hz)',ylabel='Fase (rad)',title='Fase (solo componentes significativas)')
    axes[1].set_xlim(-60,60); axes[2].set_xlim(-60,60)
    for a in axes:a.grid(alpha=.25)
    fig.suptitle(name,fontsize=16,fontweight='bold');fig.savefig(OUT/filename,dpi=170);plt.close(fig)

rect=(np.abs(t)<0.1).astype(float)
step=(t>=0).astype(float)
sine=np.sin(2*np.pi*10*t)
for x,name,filename in [(rect,'Pulso rectangular: ancho 0.20 s','01_pulso.png'),(step,'Escalón unitario observado durante 2 s','02_escalon.png'),(sine,'Senoidal: 10 Hz','03_senoidal.png')]:plot_signal(x,name,filename)

# Propiedad 1: linealidad F{a*x+b*y}=aX+bY
x=rect;y=sine;a=2.;b=.5
lin_error=np.max(np.abs(ft(a*x+b*y)-(a*ft(x)+b*ft(y))))
# Propiedad 2: desplazamiento temporal sin envolvimiento, pulso alejado de bordes
shift_s=.25; k=int(round(shift_s*fs)); shifted=np.roll(rect,k)
# Frecuencias DFT: desplazamiento circular exacto; para el pulso equivale a traslación sin recorte
shift_error=np.max(np.abs(ft(shifted)-ft(rect)*np.exp(-2j*np.pi*freq*shift_s)))
# Propiedad 3: escalamiento temporal x(a t) -> X(f/a)/|a| (ancho mitad, espectro más ancho)
rect_narrow=(np.abs(t)<0.05).astype(float)
# comprobación de área y ancho de lóbulo nulo: pulso 0.2s -> 5Hz, pulso 0.1s -> 10Hz
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
axes[0,0].plot(t,rect,label='x(t)');axes[0,0].plot(t,rect_narrow,label='x(2t)');axes[0,0].set_xlim(-.4,.4);axes[0,0].set_title('Escalamiento temporal');axes[0,0].legend()
axes[0,1].plot(freq,np.abs(ft(rect)),label='X(f)');axes[0,1].plot(freq,np.abs(ft(rect_narrow)),label='F{x(2t)}');axes[0,1].set_xlim(-35,35);axes[0,1].set_title('Compresión temporal: espectro más ancho');axes[0,1].legend()
axes[1,0].plot(t,rect,label='Original');axes[1,0].plot(t,shifted,label='Retardo 0.25 s');axes[1,0].set_xlim(-.25,.55);axes[1,0].set_title('Desplazamiento temporal');axes[1,0].legend()
ph=np.angle(ft(shifted)/np.where(np.abs(ft(rect))>1e-6,ft(rect),np.nan+0j))
axes[1,1].plot(freq,ph,'.',ms=3);axes[1,1].set_xlim(-20,20);axes[1,1].set_title('Diferencia de fase por desplazamiento');axes[1,1].set_ylabel('rad')
for ax in axes.flat:ax.grid(alpha=.25);ax.set_xlabel('Tiempo (s)' if ax in (axes[0,0],axes[1,0]) else 'Frecuencia (Hz)')
fig.savefig(OUT/'04_propiedades.png',dpi=170);plt.close(fig)

# Aplicación: señal de sensores contaminada por interferencia 50 Hz; filtrado FFT
signal=np.sin(2*np.pi*8*t)+0.6*np.sin(2*np.pi*50*t)
S=np.fft.fft(signal); f=np.fft.fftfreq(N,dt); filtered=np.fft.ifft(S*(np.abs(f)<20)).real
fig,axes=plt.subplots(2,1,figsize=(10,6),layout='constrained')
axes[0].plot(t,signal,label='Señal con interferencia 50 Hz',alpha=.7);axes[0].plot(t,filtered,label='Filtrada (<20 Hz)',lw=2);axes[0].set_xlim(-.25,.25);axes[0].set_ylabel('Amplitud');axes[0].legend();axes[0].grid(alpha=.25)
axes[1].plot(freq,np.abs(ft(signal)),label='Antes');axes[1].plot(freq,np.abs(ft(filtered)),label='Después');axes[1].set_xlim(0,75);axes[1].set(xlabel='Frecuencia (Hz)',ylabel='|X(f)|');axes[1].legend();axes[1].grid(alpha=.25)
fig.suptitle('Aplicación: supresión de interferencia de 50 Hz');fig.savefig(OUT/'05_aplicacion.png',dpi=170);plt.close(fig)
print('Error linealidad:',lin_error)
print('Error desplazamiento:',shift_error)
print('Picos senoidal: +/-10 Hz; rectángulo: primeros ceros +/-5 Hz')
