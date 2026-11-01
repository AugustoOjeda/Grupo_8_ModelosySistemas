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
C_inicialesbis=[0,-500,0]
C_inicialesbis2=[15,-700,5]
C_Inicialeseq2_0=[-30, 700, 1]
C_Inicialeseq2_1=[-15, -399, 0]
C_Inicialeseq2_2=[-30, -700, -18]
u=0.2 #mg min^-2




# Puntos de equilibrio
#Igualando las derivadas a 0:
#Necesito definir puntos iniciales
Solucion_1 = root(lambda y: modelo(0, y), C_iniciales) #Root me pide que le un punto para arrancar. por eso meto C iniciales y escribo lambda y: modelo(0, y) asi entiende que quiere variar y obtener valores de M C Y S 
Solucion_2 = root(lambda y: modelo(0, y), [-16, 800, 2])
print("=== PUNTOS DE EQUILIBRIO ===")
print("Solución 1:", Solucion_1.x)
print("Solución 2:", Solucion_2.x)
#Variables de simulacion
N=5000
t_max=100
t=np.linspace(0,t_max, N) #array de tiempo a paso de 0,02


#Aplico ode para el primer punto de equilibrio
sol = solve_ivp(modelo, (0, t_max), C_iniciales, t_eval=t) #En las condiciones inciales de la catedra

sol2 = solve_ivp(modelo, (0, t_max), C_inicialesbis, t_eval=t)# En condiciones cercanas a las iniciales por debajo

sol3 = solve_ivp(modelo, (0, t_max), C_inicialesbis2, t_eval=t) #En condiciones cercanas a las iniciales por encima

#Aplico ode para el primer punto de equilibrio

sol4= solve_ivp(modelo, (0, t_max), C_Inicialeseq2_0, t_eval=t)# En condiciones cercanas a al segundo punto de equilibrio 
sol5= solve_ivp(modelo, (0, t_max), C_Inicialeseq2_1, t_eval=t)# En condiciones cercanas a al segundo punto de equilibrio
sol6= solve_ivp(modelo, (0, t_max), C_Inicialeseq2_2, t_eval=t)# En condiciones cercanas a al segundo punto de equilibrio

# ==========================================
# Gráficas - Soluciones Temporales
# ==========================================

# Vamos a graficar las soluciones temporales solo para la condición inicial del enunciado (ci_1)
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='C (Concentración)', color='limegreen', linewidth=2)
plt.plot(sol.t, sol.y[1], label='M (Masa acumulada)', color='seagreen', linewidth=2)
plt.plot(sol.t, sol.y[2], label='S (Fuente)', color='darkgreen', linewidth=2)
# Graficamos el primer punto de equilibrio
plt.axhline(Solucion_1.x[0], color='b', linestyle='--', alpha=0.5, label='Equilibrio 1 C')
plt.axhline(Solucion_1.x[1], color='g', linestyle='--', alpha=0.5, label='Equilibrio 1 M')
plt.axhline(Solucion_1.x[2], color='r', linestyle='--', alpha=0.5, label='Equilibrio 1 S')

# Graficamos el segundo punto de equilibrio
plt.axhline(Solucion_2.x[0], color='b', linestyle=':', alpha=0.5, label='Equilibrio 2 C')
plt.axhline(Solucion_2.x[1], color='g', linestyle=':', alpha=0.5, label='Equilibrio 2 M')
plt.axhline(Solucion_2.x[2], color='r', linestyle=':', alpha=0.5, label='Equilibrio 2 S')
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

fig = plt.figure(figsize=(15, 8))
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
    label='Trayectoria 3 (CI=[15,-700,5])'
)

# Para el segundo punto de equlibrio

ax.plot(
    sol4.y[0], sol4.y[1], sol4.y[2],
    color='orchid',
    label='Trayectoria 4 (CI=[-30, 700, 1])'
)

ax.plot(
    sol5.y[0], sol5.y[1], sol5.y[2],
    color='purple',
    label='Trayectoria 5 (CI=[-15, -399, 0])'
)

ax.plot(
    sol6.y[0], sol6.y[1], sol6.y[2],
    color='darkmagenta',
    label='Trayectoria 6 (CI=[-30, -700, 18])'
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
    label='Inicio 3'
)

ax.scatter(
    -30,700,1,
    color='orchid',
    marker='o',
    s=50,
    label='Inicio 4'
)

ax.scatter(
    -15,-399,0,
    color='purple',
    marker='o',
    s=50,
    label='Inicio 5'
)

ax.scatter(
    -30,-700,-18,
    color='darkmagenta',
    marker='o',
    s=50,
    label='Inicio 6'
)

# Punto de equilibrio
ax.scatter(
    Solucion_1.x[0], Solucion_1.x[1], Solucion_1.x[2],
    color='k',
    marker='X',
    s=150,
    label='Punto de Equilibrio 16.39352298 -769.67614889    2'
)

ax.scatter(
    Solucion_2.x[0], Solucion_2.x[1], Solucion_2.x[2],
    color='k',
    marker='X',
    s=150,
    label='Punto de Equilibrio -15.71352298 835.67614889   2'
)

ax.set_xlabel('C (Concentración)')
ax.set_ylabel('M (Masa Acumulada)')
ax.set_zlabel('S (Fuente)')
ax.set_title('Diagrama de Fase 3D')
ax.legend(
    loc='center left',
    bbox_to_anchor=(1.05, 0.5),
    fontsize=9,
    frameon=True
)

fig.subplots_adjust(right=0.70)
plt.tight_layout()
plt.show()


