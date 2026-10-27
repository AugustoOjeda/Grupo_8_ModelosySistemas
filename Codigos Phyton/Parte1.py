# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:50:22 2026

@author: Augusto
"""
#Librerias a usar 

import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp


# ================================
#           Grupo 8
# ================================

#Modelo de reactor dado

def modelo(t,y):
    C,M,S=y #defino C M y S como vector dependiente del tiempo
    
    C_der=-(Q/V)*C-K*C+S
    M_der=Q*(Cin-C)-kr*M
    S_der=-Lamda*S+u
    return[C_der,M_der,S_der]


#CTES
Q= 1 #L min^-1
Cin= 1 #mgL^-1
kr= 0.02 #min^-1
V= 10 #L
K=0.05 #min^-1
Lamda=0.1 #min^-1

#Condiciones iniciales
C_0=0.5 #mg L^-1
M_0=1 #mg
S_0=0.5 #mg L^-1 min^-1
C_iniciales=[C_0,M_0,S_0]
C_inicialesbis=[0,-500,0]
C_inicialesbis2=[15,-700,5]
u=0.2 #mg min^-2




# Puntos de equilibrio

#Igualando las derivadas a 0:
#dS/dt = 0 -> -lam * S_eq + u = 0
S_eq = u / Lamda

#dC/dt = 0 -> -(Q/V)C_eq - kC_eq + S_eq = 0
C_eq = S_eq / ((Q/V) + K)

#dM/dt = 0 -> Q(Cin - C_eq) - krM_eq = 0
M_eq = (Q * (Cin - C_eq)) / kr

print("=== PUNTOS DE EQUILIBRIO ===")
print(f"C_eq = {C_eq:.4f} mg L^-1")
print(f"M_eq = {M_eq:.4f} mg")
print(f"S_eq = {S_eq:.4f} mg L^-1 min^-1\n")

#Variables de simulacion
N=5000
t_max=100
t=np.linspace(0,t_max, N) #array de tiempo a paso de 0,02


#Aplico ode
sol = solve_ivp(modelo, (0, t_max), C_iniciales, t_eval=t) #En las condiciones inciales de la catedra

sol2 = solve_ivp(modelo, (0, t_max), C_inicialesbis, t_eval=t)# En condiciones cercanas a las iniciales por debajo

sol3 = solve_ivp(modelo, (0, t_max), C_inicialesbis2, t_eval=t) #En condiciones cercanas a las iniciales por encima

# ==========================================
# Gráficas - Soluciones Temporales
# ==========================================

# Vamos a graficar las soluciones temporales solo para la condición inicial del enunciado (ci_1)
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='C (Concentración)', color='limegreen', linewidth=2)
plt.plot(sol.t, sol.y[1], label='M (Masa acumulada)', color='seagreen', linewidth=2)
plt.plot(sol.t, sol.y[2], label='S (Fuente)', color='darkgreen', linewidth=2)
plt.axhline(C_eq, color='b', linestyle='--', alpha=0.5, label='Equilibrio C')
plt.axhline(M_eq, color='g', linestyle='--', alpha=0.5, label='Equilibrio M')
plt.axhline(S_eq, color='r', linestyle='--', alpha=0.5, label='Equilibrio S')
plt.plot(sol.t, sol2.y[0], label='C2 (Concentración)', color='indigo', linewidth=2)
plt.plot(sol.t, sol2.y[1], label='M2 (Masa acumulada)', color='darkmagenta', linewidth=2)
plt.plot(sol.t, sol2.y[2], label='S2 (Fuente)', color='orchid', linewidth=2)

plt.plot(sol.t, sol3.y[0], label='C3 (Concentración)', color='turquoise', linewidth=2)
plt.plot(sol.t, sol3.y[1], label='M3 (Masa acumulada)', color='aquamarine', linewidth=2)
plt.plot(sol.t, sol3.y[2], label='S3 (Fuente)', color='darkcyan', linewidth=2)

plt.legend()

plt.title('Soluciones Temporales del Sistema (CI del enunciado)')
plt.xlabel('Tiempo t (min)')
plt.ylabel('Magnitud')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# Gráficas - Diagrama de Fase 3D
# ==========================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Trayectoria del sistema
ax.plot(
    sol.y[0], sol.y[1], sol.y[2],
    color='b',
    label='Trayectoria 1 (CI Enunciado)'
)
ax.plot(
    sol2.y[0], sol2.y[1], sol2.y[2],
    color='g',
    label='Trayectoria 2 (CI=[0,-500,0])'
)

ax.plot(
    sol3.y[0], sol3.y[1], sol3.y[2],
    color='r',
    label='Trayectoria 2 (CI=[15,-700,5])'
)

# Punto inicial
ax.scatter(
    C_0, M_0, S_0,
    color='b',
    marker='o',
    s=50,
    label='Inicio 1'
)

ax.scatter(
    0,-500,0,
    color='g',
    marker='o',
    s=50,
    label='Inicio 2'
)

ax.scatter(
    15,-700,5,
    color='r',
    marker='o',
    s=50,
    label='Inicio 2'
)

# Punto de equilibrio
ax.scatter(
    C_eq, M_eq, S_eq,
    color='k',
    marker='X',
    s=150,
    label='Punto de Equilibrio'
)

ax.set_xlabel('C (Concentración)')
ax.set_ylabel('M (Masa Acumulada)')
ax.set_zlabel('S (Fuente)')
ax.set_title('Diagrama de Fase 3D')
ax.legend()

plt.tight_layout()
plt.show()


