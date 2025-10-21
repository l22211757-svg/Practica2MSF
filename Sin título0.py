"""
Práctica 0: Mecánica pulmonar

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: IGonzalez Sanchez Paola 
Número de control: 22211757
Correo institucional: L22211757@tectijuana.edu.mx
Asignatura: Modelado de Sistemas Fisiologicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
# Instalar librerias en consola
#!pip install control
#!pip install slycot
import control

# Librerías para cálculo numérico y generación de gráficas
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

u = np.array(pd.read_excel('signal.xlsx',header=None))
x0,t0,tF,dt,w,h = 0,0,10,1E-3,10,5 
N= round ((tF-t0)/dt)+1
t=np.linspace(t0,tF,N)
u = np.reshape(signal.resample(u,len(t)),-1)


def cardio (Z,C,R,L):
    num= [L*R,R*Z]
    den= [C*L*R*Z,L*R+L*Z,R*Z]
    sys= control.tf(num,den)
    return sys 

# Funcion de transferencia : Individuo Normatenso(conrtrol)
Z,C,R,L=0.033,1.5,0.95,0.01
sysnormo = cardio(Z,C,R,L)
print(f'individuo normotemso (control): {sysnormo}')
# Funcion de transferencia : Individuo Hipotenso (caso)
Z,C,R,L=0.02,0.250,0.6,0.005
syshipo = cardio(Z,C,R,L)
print (f'individuo hipotenso (caso): {syshipo}')
# Funcion de transferencia : Individuo Hipertenso (caso)
Z,C,R,L=0.05,2.5,1.4,0.02
syshiper = cardio(Z,C,R,L)
print (f'individuo hipertenso (caso): {syshiper}')

clr1 = np.array([11, 29, 81])/255
clr2 = np.array([114, 94, 173])/255
clr3 = np.array([140, 205, 235])/255
clr4 = np.array([255, 227, 169])/255
clr5 = np.array([250, 129, 47])/255
clr6 = np.array([145, 18, 188])/255
clr7 = np.array([223, 57, 105])/255
clr8= np.array([5, 127, 109 ])/255

#Respuesta en lazo abierto
_,Pp0 = control.forced_response(sysnormo,t,u,x0)
_,Pp1 = control.forced_response(syshipo,t,u,x0)
_,Pp2 = control.forced_response(syshiper,t,u,x0)
fg1 = plt.figure()
plt.plot(t,Pp0,'-', linewidth=1,color=clr1,label='Pp(t) : Normotenso')
plt.plot(t,Pp1,'-', linewidth=1,color=clr2,label='Pp(t) : Hipotenso')
plt.plot(t,Pp2,'-', linewidth=1,color=clr3,label='Pp(t) : Hipertenso')
plt.grid(False)
plt.xlim(0,10);plt.xticks(np.arange(0,11,1))
plt.ylim(-0.4,1.2);plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('t [s]')
plt.ylabel('Pp (t) [V]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=3)
plt.show()
fg1.set_size_inches(w,h)
fg1.tight_layout()
fg1.savefig('sistema cardiovascular python.png',dpi=600,bbox_inches='tight')
fg1.savefig('sistema cardiovascular python.pdf',bbox_inches='tight')

def controlador (kP, kI):
    Cr = 1E-6
    Re = 1/(kI*Cr)
    Rr = kP*Re
    numPI = [Rr*Cr,1]
    denPI = [Re*Cr,0]
    PI = control.tf(numPI,denPI)
    return PI

PI = controlador(10,4003.61190219458)
X = control.series(PI,syshipo)
hipo_PI = control.feedback(X,1,sign=-1)

PI = controlador(10,4299.5839112344)
X = control.series(PI,syshipo)
hiper_PI = control.feedback(X,1,sign=-1)

_,Pp3 = control.forced_response(hipo_PI,t,Pp0,x0)
_,Pp4 = control.forced_response(hiper_PI,t,Pp0,x0)

#Respuestas en lazo cerrado
fg2 = plt.figure()
plt.plot(t,Pp0,'-', linewidth=1,color=clr1,label='Pp(t) : Normotenso')
plt.plot(t,Pp1,'-', linewidth=1,color=clr2,label='Pp(t) : Hipotenso')
plt.plot(t,Pp2,'-', linewidth=1,color=clr3,label='Pp(t) : Hipertenso')
plt.plot(t,Pp3,':', linewidth=2,color=clr4,label='Pp(t) : Hipotenso PI')
plt.plot(t,Pp4,'-.', linewidth=2,color=clr5,label='Pp(t) : Hipertenso PI')

plt.grid(False)
plt.xlim(0,10);plt.xticks(np.arange(0,11,1))
plt.ylim(-0.4,1.2);plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('t [s]')
plt.ylabel('Pp (t) [V]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=5)
plt.show()
fg2.set_size_inches(w,h)
fg2.tight_layout()
fg2.savefig('sistema cardiovascular PI python.png',dpi=600,bbox_inches='tight')
fg2.savefig('sistema cardiovascular PI python.pdf',bbox_inches='tight')
