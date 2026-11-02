# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:50:22 2026

@author: Augusto
"""
#Librerias a usar 

import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from scipy.optimize import root

# ================================
#           Grupo 8
# ================================

#Modelo de reactor dado

def modelo(t,y):
    C,M,S=y #defino C M y S como vector dependiente del tiempo
    
    C_der=-(Q/V)*C-(Kmax*C/(Km+C))+S
    M_der=Q*(Cin-C)-kr*M
    S_der=-Lamda*S+u
    return[C_der,M_der,S_der]

def jacobiano_Modelo(y):
   C,M,S=y
   A=np.array([[-Q/V-(Kmax*Km)/(Km+C)**2,0,1],
              [-Q, -kr, 0],
              [0, 0, -Lamda]])
    
    
   return A



#CTES
Q= 1 #L min^-1
Cin= 1 #mgL^-1
kr= 0.02 #min^-1
V= 10 #L
K=0.05 #min^-1
Lamda=0.1 #min^-1
Km=12.88 #mg L^-1
Kmax=K * Km #mg L^-1 min^-1


#Condiciones iniciales
C_0=0.5 #mg L^-1
M_0=1 #mg
S_0=0.5 #mg L^-1 min^-1
C_iniciales=[C_0,M_0,S_0]
u=0.2 #mg min^-2




# Puntos de equilibrio
Solucion_1 = root(lambda y: modelo(0, y), C_iniciales) #Los dos puntos de equlibrio me dieron masas negativas. Revisar, ya venia desde el modelo lineal ese problema
print("=== PUNTOS DE EQUILIBRIO ESCOGIDO===")
print("Solución 1:", Solucion_1.x)
X_eq = Solucion_1.x #esto genera el array X de equilibrio se podria no hacer pero creo que queda mas comodo para la sintaxis de quien lo lee

#Variables de simulacion
N=5000
t_max=100
t=np.linspace(0,t_max, N) #array de tiempo a paso de 0,02


#Evaluo el jacobiano en el punto de equlibrio
A1=jacobiano_Modelo(X_eq)
print("Jacobiano evaluado en el punto")
print(A1)

print("\n=== ECUACIONES LINEALIZADAS ===")

print(f"C_der = {A1[0,0]:.5f}*(C - ({X_eq[0]:.5f})) + (S - ({X_eq[2]:.5f}))")

print(f"M_der = {A1[1,0]:.5f}*(C - ({X_eq[0]:.5f})) + {A1[1,1]:.5f}*(M - ({X_eq[1]:.5f}))")

print(f"S_der = {A1[2,2]:.5f}*(S - ({X_eq[2]:.5f}))")


#Quedaria calcular los Autovalores para este punto de equlibrio asi decidimos si usar este punto de equilibrio o otro 
#Para calcular los autovalores hay que usar np.linalg.eigvals(A1)
#Eso literalmente te calcula el policaracteristico y resuelve la ecuacion igualando a 0
#Devuelve un vector Solucion1=[ava.0, ava.1, ava.2]
#Para usarlo dsp podemos printear como Ava1=Solucion1.0,Ava2=Solucion1.1, Ava3=Solucion1.2
#Si eso nos da bien y el grafico al punto de equilibrio converge le mandamso mail a la profe y si da el ok mañana terminamos ya lo de modelos.
#Si mañana terminamos el codigo el finde le damos al informe y ya podemos entregar una version 1 a ver si paula nos dice mandale para adelante 
#Ademas capas no tenemos que ni anotarnos los viernes 



