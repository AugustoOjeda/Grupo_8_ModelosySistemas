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
from scipy.signal import TransferFunction, impulse, step
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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


def modelo_linealizado(t,y,A,X_eq):
    X_linealizado=A@(y-X_eq)
    return X_linealizado


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
C_iniciales=[C_0,M_0,S_0]
C_inicialesbis=[0,-500,0]
C_inicialesbis2=[15,-700,5]
C_Inicialeseq2_0=[-30, 700, 1]
C_Inicialeseq2_1=[-15, -399, 0]
C_Inicialeseq2_2=[-30, -700, -18]



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



#Calculo de autovalores
autovalores=np.linalg.eigvals(A1)
print("\n=== Calculo de autovalores ===")
print(f"λ1 = {autovalores[0]}, λ2 = {autovalores[1]}, λ3 = {autovalores[2]}")


# Diagrama de fase 3D - Nodo sumidero

# Tiempo de simulación
t_fase = np.linspace(0, 300, 1000)

# Escalas para normalizar las variables
escalas = np.array([1, 50, 0.2])

# Generamos muchas perturbaciones iniciales
rng = np.random.default_rng(8)
N_trayectorias = 100
perturbaciones = rng.uniform(-1, 1, size=(N_trayectorias, 3)) * escalas

# Colores para las trayectorias
colores = plt.cm.viridis(np.linspace(0, 1, N_trayectorias))

# Creamos figura con 3 subgráficos
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Pares de variables a proyectar
# (indice_x, indice_y, nombre_x, nombre_y)
proyecciones = [
    (0, 1, r'$\Delta C$ / (1 mg/L)', r'$\Delta M$ / (50 mg)'),
    (0, 2, r'$\Delta C$ / (1 mg/L)', r'$\Delta S$ / (0.2 mg/(L min))'),
    (1, 2, r'$\Delta M$ / (50 mg)', r'$\Delta S$ / (0.2 mg/(L min))')
]


# Campo de direcciones 2D

grid_vals = np.linspace(-1.2, 1.2, 15)

for ax, (ix, iy, xlabel, ylabel) in zip(axes, proyecciones):

    # Malla para el campo
    Xg, Yg = np.meshgrid(grid_vals, grid_vals)

    Ug = np.zeros_like(Xg)
    Vg = np.zeros_like(Yg)

    # Construimos el campo a partir del modelo linealizado
    for m in range(Xg.shape[0]):
        for n in range(Xg.shape[1]):

            # Estado desplazado normalizado -> lo llevamos a unidades reales
            delta = np.zeros(3)
            delta[ix] = Xg[m, n] * escalas[ix]
            delta[iy] = Yg[m, n] * escalas[iy]

            # En el sistema linealizado: dX = A1 @ delta
            dX = A1 @ delta

            # Normalizamos la derivada en los ejes elegidos
            Ug[m, n] = dX[ix] / escalas[ix]
            Vg[m, n] = dX[iy] / escalas[iy]

    # Dibujamos campo de direcciones
    ax.quiver(Xg, Yg, Ug, Vg, color='gray', alpha=0.5)

# Trayectorias

    for i, perturbacion in enumerate(perturbaciones):

        # Condición inicial cerca del equilibrio
        X0 = X_eq + perturbacion

        # Resolvemos el sistema linealizado
        sol_lin = solve_ivp(
            modelo_linealizado,
            (0, 300),
            X0,
            t_eval=t_fase,
            args=(A1, X_eq)
        )

        # Variables desplazadas y normalizadas
        Z = (sol_lin.y - X_eq[:, None]) / escalas[:, None]

        # Dibujamos la proyección 2D
        ax.plot(
            Z[ix], Z[iy],
            color=colores[i],
            linewidth=1,
            alpha=0.6
        )

        # Marcamos el punto inicial
        ax.scatter(
            Z[ix, 0], Z[iy, 0],
            color=colores[i],
            s=10,
            alpha=0.8
        )

    # Punto de equilibrio
    ax.scatter(
        0, 0,
        color='black',
        marker='X',
        s=120,
        label='Equilibrio'
    )

    ax.set_title(f'Retrato de fase: {xlabel} vs {ylabel}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

plt.suptitle('Retratos de fase 2D - Sistema linealizado (Nodo sumidero)', fontsize=15)
plt.tight_layout()
plt.show()

#Analizis del punto 1 para modelo linealizado. 

# Condición inicial del enunciado
sol1_lin = solve_ivp(modelo_linealizado, (0, t_max),
                     C_iniciales, t_eval=t, args=(A1, X_eq))

# Condiciones iniciales alternativas
sol2_lin = solve_ivp(modelo_linealizado, (0, t_max),
                     C_inicialesbis, t_eval=t, args=(A1, X_eq))

sol3_lin = solve_ivp(modelo_linealizado, (0, t_max),
                     C_inicialesbis2, t_eval=t, args=(A1, X_eq))


# Gráficas - Diagrama de Fase 3D del modelo linealizado


fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection='3d')

# Trayectoria 1 - Condiciones iniciales del enunciado
ax.plot(
    sol1_lin.y[0], sol1_lin.y[1], sol1_lin.y[2],
    color='b',
    label='Trayectoria 1 (CI Enunciado)'
)

# Trayectoria 2 - Condiciones iniciales alternativas
ax.plot(
    sol2_lin.y[0], sol2_lin.y[1], sol2_lin.y[2],
    color='g',
    label=f'Trayectoria 2 (CI={C_inicialesbis})'
)

# Trayectoria 3 - Condiciones iniciales alternativas
ax.plot(
    sol3_lin.y[0], sol3_lin.y[1], sol3_lin.y[2],
    color='r',
    label=f'Trayectoria 3 (CI={C_inicialesbis2})'
)


# Puntos iniciales


ax.scatter(
    C_iniciales[0], C_iniciales[1], C_iniciales[2],
    color='b',
    marker='o',
    s=50,
    label='Inicio 1'
)

ax.scatter(
    C_inicialesbis[0], C_inicialesbis[1], C_inicialesbis[2],
    color='g',
    marker='o',
    s=50,
    label='Inicio 2'
)

ax.scatter(
    C_inicialesbis2[0], C_inicialesbis2[1], C_inicialesbis2[2],
    color='r',
    marker='o',
    s=50,
    label='Inicio 3'
)


# Punto de equilibrio


# El equilibrio del modelo linealizado es el mismo
# punto alrededor del cual calculamos el Jacobiano

ax.scatter(
    X_eq[0], X_eq[1], X_eq[2],
    color='k',
    marker='X',
    s=150,
    label=f'Punto de Equilibrio {np.round(X_eq, 4)}'
)

# Punto final trayectoria 1
ax.scatter(
    sol1_lin.y[0, -1], sol1_lin.y[1, -1], sol1_lin.y[2, -1],
    color='b',
    marker='s',
    s=70,
    label='Final 1'
)

# Punto final trayectoria 2
ax.scatter(
    sol2_lin.y[0, -1], sol2_lin.y[1, -1], sol2_lin.y[2, -1],
    color='g',
    marker='s',
    s=70,
    label='Final 2'
)

# Punto final trayectoria 3
ax.scatter(
    sol3_lin.y[0, -1], sol3_lin.y[1, -1], sol3_lin.y[2, -1],
    color='r',
    marker='s',
    s=70,
    label='Final 3'
)


# Configuración del gráfico


ax.set_xlabel('C (Concentración)')
ax.set_ylabel('M (Masa acumulada)')
ax.set_zlabel('S (Fuente)')

ax.set_title('Diagrama de Fase 3D - Modelo Linealizado')

# Leyenda fuera del gráfico
# Leyenda en un recuadro debajo del gráfico
ax.legend(
    loc='center left',
    bbox_to_anchor=(1.05, 0.5),
    fontsize=9,
    frameon=True
)

fig.subplots_adjust(right=0.70)

plt.show()

# Gráficas - Soluciones Temporales del modelo linealizado 

# Agrupamos las soluciones del modelo linealizado
soluciones = [sol1_lin, sol2_lin, sol3_lin]

# Condiciones iniciales utilizadas
condiciones = [C_iniciales, C_inicialesbis, C_inicialesbis2]

# Colores de cada trayectoria
colores = ['b', 'g', 'r']

# Variables de estado
variables = [
    'C (Concentración, mg/L)',
    'M (Masa acumulada, mg)',
    'S (Fuente, mg/(L min))'
]

# Creamos tres gráficos
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for j in range(3):

    for i in range(3):

        axes[j].plot(
            t,
            soluciones[i].y[j],
            color=colores[i],
            linewidth=2,
            label=f'CI {i+1} = {condiciones[i]}'
        )

    # Marcamos el valor de equilibrio
    axes[j].axhline(
        X_eq[j],
        color='k',
        linestyle='--',
        alpha=0.7,
        label='Equilibrio'
    )

    axes[j].set_ylabel(variables[j])
    axes[j].grid(True, alpha=0.3)
    axes[j].legend(fontsize=9)

# Configuración general
axes[2].set_xlabel('Tiempo (min)')

fig.suptitle(
    'Soluciones Temporales - Modelo Linealizado',
    fontsize=15
)

plt.tight_layout()
plt.show()

# ==========================================
# Polos y ceros de la función de transferencia
# Entrada: u(t) - Salida: C(t)
# ==========================================

# Coeficientes obtenidos del Jacobiano
a = -A1[0, 0]
lam = -A1[2, 2]

# Función de transferencia G(s) = num(s)/den(s)
numerador = np.array([A1[0, 2]])

# Multiplicamos (s+a)*(s+lam)
denominador = np.polymul([1, a], [1, lam])


# Calculamos las raíces
polos = np.roots(denominador)
ceros = np.roots(numerador)

print("\n=== POLOS Y CEROS ===")
print("Numerador:", numerador)
print("Denominador:", denominador)
print("Polos:", polos)
print("Ceros:", ceros)


# ==========================================
# Mapa de Polos y Ceros (Plano S)
# con recuadro de zoom sobre los polos
# ==========================================

fig2, ax3 = plt.subplots(figsize=(8, 6))

# Dibujamos el círculo unitario punteado al fondo (zorder=1)
theta = np.linspace(0, 2 * np.pi, 200)
ax3.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--',
         label='Círculo unitario', zorder=1)

# Dibujamos los ejes cartesianos al fondo (zorder=1)
ax3.axhline(0, color='gray', linewidth=0.8, zorder=1)
ax3.axvline(0, color='gray', linewidth=0.8, zorder=1)

# Graficamos los Ceros adelante (zorder=2)
if len(ceros) > 0:
    ax3.scatter(np.real(ceros), np.imag(ceros),
                s=80, facecolors='none',
                edgecolors='blue', linewidth=2,
                label='Ceros (o)', zorder=2)

# Graficamos los Polos al frente de todo (zorder=3)
if len(polos) > 0:
    ax3.scatter(np.real(polos), np.imag(polos),
                s=120, marker='x',
                color='orange', linewidth=3,
                label='Polos (x)', zorder=3)

# Ajustes estéticos del gráfico principal
ax3.set_title('Mapa de Polos y Ceros (Plano S)')
ax3.set_xlabel(r'$\Re(s)$ (Parte Real)')
ax3.set_ylabel(r'$\Im(s)$ (Parte Imaginaria)')
ax3.grid(True, which='both', linestyle=':', alpha=0.6, zorder=1)

# Relación de aspecto 1:1 para que el círculo no se vea ovalado
ax3.axis('equal')
ax3.set_xlim([-1.5, 1.5])
ax3.set_ylim([-1.5, 1.5])
ax3.legend(loc='upper left')


# ==========================================
# Recuadro de zoom sobre los polos
# ==========================================

# Creamos un inset (recuadro pequeño)
axins = inset_axes(ax3, width="38%", height="38%", loc='lower right')

# Mismo contenido en el inset
axins.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', zorder=1)
axins.axhline(0, color='gray', linewidth=0.8, zorder=1)
axins.axvline(0, color='gray', linewidth=0.8, zorder=1)

if len(ceros) > 0:
    axins.scatter(np.real(ceros), np.imag(ceros),
                  s=80, facecolors='none',
                  edgecolors='blue', linewidth=2, zorder=2)

if len(polos) > 0:
    axins.scatter(np.real(polos), np.imag(polos),
                  s=120, marker='x',
                  color='orange', linewidth=3, zorder=3)

# Elegimos la región de zoom alrededor de los polos
xmin = np.min(np.real(polos)) - 0.005
xmax = np.max(np.real(polos)) + 0.005
ymin = -0.01
ymax = 0.01

axins.set_xlim(xmin, xmax)
axins.set_ylim(ymin, ymax)

# Estética del recuadro
axins.grid(True, linestyle=':', alpha=0.6)
axins.set_title('Zoom polos', fontsize=9)


plt.tight_layout()
plt.show()

# Respuesta al impulso y al escalon

G = TransferFunction(numerador, denominador)

# Calculamos la respuesta al impulso
t_impulso, y_impulso = impulse(G, T=t)

# Calculamos la respuesta al escalón
t_escalon, y_escalon = step(G, T=t)

# ==========================================
# Gráficas de las respuestas temporales
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Respuesta al impulso
axes[0].plot(
    t_impulso, y_impulso,
    color='blue',
    linewidth=2
)

axes[0].set_title('Respuesta al Impulso')
axes[0].set_xlabel('Tiempo (min)')
axes[0].set_ylabel('Respuesta de C')
axes[0].grid(True, alpha=0.3)

# Respuesta al escalón
axes[1].plot(
    t_escalon, y_escalon,
    color='red',
    linewidth=2
)

axes[1].set_title('Respuesta al Escalón')
axes[1].set_xlabel('Tiempo (min)')
axes[1].set_ylabel('Respuesta de C')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Respuestas Temporales del Sistema Linealizado')
plt.tight_layout()
plt.show()

