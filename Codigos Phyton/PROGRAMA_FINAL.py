# -*- coding: utf-8 -*-
"""
Universidad Nacional de San Martin
Modelos y Sistemas - Modelado de contaminate ambientan de recursos hidricos

BALLAN - OJEDA - PANTOZZI

"""

#Librerias a usar 

import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.signal import TransferFunction, impulse, step
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ====================================
# ======  Definicion de modelos ======
# ====================================

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
C_inicialesbis=[30,500,10]
C_inicialesbis2=[15,700,25]
C_Inicialeseq2_0=[40, 700, 1]
C_Inicialeseq2_1=[150, 1500, 0]
C_Inicialeseq2_2=[10, 10, 18]
u=0.2 #mg min^-2


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

# ====================================
# ======  Puntos de Equilibrio  ======
# ====================================

Solucion_1 = root(lambda y: modelo(0, y), C_iniciales) #Root me pide que le un punto para arrancar. por eso meto C iniciales y escribo lambda y: modelo(0, y) asi entiende que quiere variar y obtener valores de M C Y S 
Solucion_2 = root(lambda y: modelo(0, y), [-16, 800, 2])
X_eq = Solucion_1.x
print("=== PUNTOS DE EQUILIBRIO ===")
print("P.E. 1:", Solucion_1.x)
print("P.E. 2:", Solucion_2.x)

# ====================================
# ======  Analisis temporales  =======
# ====================================

#Variables de simulacion
N=5000
t_max=100
t=np.linspace(0,t_max, N)

#Aplico ode para el primer punto de equilibrio
sol = solve_ivp(modelo, (0, t_max), C_iniciales, t_eval=t) 
sol2 = solve_ivp(modelo, (0, t_max), C_inicialesbis, t_eval=t)# En condiciones cercanas a las iniciales por debajo
sol3 = solve_ivp(modelo, (0, t_max), C_inicialesbis2, t_eval=t) #En condiciones cercanas a las iniciales por encima
#Aplico ode para el primer punto de equilibrio
sol4= solve_ivp(modelo, (0, t_max), C_Inicialeseq2_0, t_eval=t)# En condiciones cercanas a al segundo punto de equilibrio 
sol5= solve_ivp(modelo, (0, t_max), C_Inicialeseq2_1, t_eval=t)# En condiciones cercanas a al segundo punto de equilibrio
sol6= solve_ivp(modelo, (0, t_max), C_Inicialeseq2_2, t_eval=t)# En condiciones cercanas a al segundo punto de equilibrio

# ==============================
# ======  Linealizacion  =======
# ==============================

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

#Analizis del punto 1 para modelo linealizado. 
# Condición inicial del enunciado
sol1_lin = solve_ivp(modelo_linealizado, (0, t_max),
                     C_iniciales, t_eval=t, args=(A1, X_eq))
# Condiciones iniciales alternativas
sol2_lin = solve_ivp(modelo_linealizado, (0, t_max),
                     C_inicialesbis, t_eval=t, args=(A1, X_eq))
sol3_lin = solve_ivp(modelo_linealizado, (0, t_max),
                     C_inicialesbis2, t_eval=t, args=(A1, X_eq))

#===========================================
#======= Autovalores Linealizado ===========
#===========================================

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




# ==========================================
# COMPARACIÓN DE SOLUCIONES TEMPORALES
# Modelo no lineal vs modelo linealizado
# ==========================================

fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col')

# ------------------------------
# COLUMNA IZQUIERDA: NO LINEAL
# ------------------------------

# C(t)
axes[0, 0].plot(sol.t,  sol.y[0],  label='Trayectoria 1', color='limegreen', linewidth=2)
axes[0, 0].plot(sol2.t, sol2.y[0], label='Trayectoria 2', color='indigo', linewidth=2)
axes[0, 0].plot(sol3.t, sol3.y[0], label='Trayectoria 3', color='turquoise', linewidth=2)
axes[0, 0].axhline(Solucion_1.x[0], color='k', linestyle='--', alpha=0.7, label='Equilibrio')
axes[0, 0].set_ylabel('C (Concentración, mg/L)')
axes[0, 0].grid(True)
axes[0, 0].legend(fontsize=8)

# M(t)
axes[1, 0].plot(sol.t,  sol.y[1],  label='Trayectoria 1', color='seagreen', linewidth=2)
axes[1, 0].plot(sol2.t, sol2.y[1], label='Trayectoria 2', color='darkmagenta', linewidth=2)
axes[1, 0].plot(sol3.t, sol3.y[1], label='Trayectoria 3', color='aquamarine', linewidth=2)
axes[1, 0].axhline(Solucion_1.x[1], color='k', linestyle='--', alpha=0.7, label='Equilibrio')
axes[1, 0].set_ylabel('M (Masa acumulada, mg)')
axes[1, 0].grid(True)
axes[1, 0].legend(fontsize=8)

# S(t)
axes[2, 0].plot(sol.t,  sol.y[2],  label='Trayectoria 1', color='darkgreen', linewidth=2)
axes[2, 0].plot(sol2.t, sol2.y[2], label='Trayectoria 2', color='orchid', linewidth=2)
axes[2, 0].plot(sol3.t, sol3.y[2], label='Trayectoria 3', color='darkcyan', linewidth=2)
axes[2, 0].axhline(Solucion_1.x[2], color='k', linestyle='--', alpha=0.7, label='Equilibrio')
axes[2, 0].set_xlabel('Tiempo (min)')
axes[2, 0].set_ylabel('S (Fuente, mg/(L min))')
axes[2, 0].grid(True)
axes[2, 0].legend(fontsize=8)

# ------------------------------
# COLUMNA DERECHA: LINEALIZADO
# ------------------------------

# C(t)
axes[0, 1].plot(t, sol1_lin.y[0], label='Trayectoria 1', color='b', linewidth=2)
axes[0, 1].plot(t, sol2_lin.y[0], label='Trayectoria 2', color='g', linewidth=2)
axes[0, 1].plot(t, sol3_lin.y[0], label='Trayectoria 3', color='r', linewidth=2)
axes[0, 1].axhline(X_eq[0], color='k', linestyle='--', alpha=0.7, label='Equilibrio')
axes[0, 1].set_ylabel('C (Concentración, mg/L)')
axes[0, 1].grid(True)
axes[0, 1].legend(fontsize=8)

# M(t)
axes[1, 1].plot(t, sol1_lin.y[1], label='Trayectoria 1', color='b', linewidth=2)
axes[1, 1].plot(t, sol2_lin.y[1], label='Trayectoria 2', color='g', linewidth=2)
axes[1, 1].plot(t, sol3_lin.y[1], label='Trayectoria 3', color='r', linewidth=2)
axes[1, 1].axhline(X_eq[1], color='k', linestyle='--', alpha=0.7, label='Equilibrio')
axes[1, 1].set_ylabel('M (Masa acumulada, mg)')
axes[1, 1].grid(True)
axes[1, 1].legend(fontsize=8)

# S(t)
axes[2, 1].plot(t, sol1_lin.y[2], label='Trayectoria 1', color='b', linewidth=2)
axes[2, 1].plot(t, sol2_lin.y[2], label='Trayectoria 2', color='g', linewidth=2)
axes[2, 1].plot(t, sol3_lin.y[2], label='Trayectoria 3', color='r', linewidth=2)
axes[2, 1].axhline(X_eq[2], color='k', linestyle='--', alpha=0.7, label='Equilibrio')
axes[2, 1].set_xlabel('Tiempo (min)')
axes[2, 1].set_ylabel('S (Fuente, mg/(L min))')
axes[2, 1].grid(True)
axes[2, 1].legend(fontsize=8)

plt.tight_layout()
fig.subplots_adjust(bottom=0.14)

# Título debajo de la columna izquierda
fig.text(
    0.27, 0.035,
    'Figura 1: Soluciones temporales - Modelo no lineal',
    ha='center',
    va='center',
    fontsize=11,
    fontweight='bold'
)

# Título debajo de la columna derecha
fig.text(
    0.76, 0.035,
    'Figura 2: Soluciones temporales - Modelo linealizado al rededor del punto [16.393 -769.676  2]',
    ha='center',
    va='center',
    fontsize=11,
    fontweight='bold'
)

plt.show()


# ==========================================
# COMPARACIÓN DE DIAGRAMAS DE FASE 3D
# Modelo no lineal vs modelo linealizado
# ==========================================

fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')


# ------------------------------
# IZQUIERDA: MODELO NO LINEAL
# ------------------------------

# Trayectorias del sistema
ax1.plot(sol.y[0], sol.y[1], sol.y[2], color='b',
         label='Trayectoria 1 (CI Enunciado)')

ax1.plot(sol2.y[0], sol2.y[1], sol2.y[2], color='g',
         label='Trayectoria 2 (CI=[30,500,10])')

ax1.plot(sol3.y[0], sol3.y[1], sol3.y[2], color='r',
         label='Trayectoria 3 (CI=[15,700,25])')

ax1.plot(sol4.y[0], sol4.y[1], sol4.y[2], color='orchid',
         label='Trayectoria 4 (CI=[40, 700, 1])')

ax1.plot(sol5.y[0], sol5.y[1], sol5.y[2], color='navy',
         label='Trayectoria 5 (CI=[150, 1500, 0])')

ax1.plot(sol6.y[0], sol6.y[1], sol6.y[2], color='darkmagenta',
         label='Trayectoria 6 (CI=[10, 10, 18])')


# ------------------------------
# Puntos iniciales
# ------------------------------

ax1.scatter(C_iniciales[0], C_iniciales[1], C_iniciales[2],
            color='b', marker='o', s=50, label='Inicio 1')

ax1.scatter(C_inicialesbis[0], C_inicialesbis[1], C_inicialesbis[2],
            color='g', marker='o', s=50, label='Inicio 2')

ax1.scatter(C_inicialesbis2[0], C_inicialesbis2[1], C_inicialesbis2[2],
            color='r', marker='o', s=50, label='Inicio 3')

ax1.scatter(C_Inicialeseq2_0[0], C_Inicialeseq2_0[1], C_Inicialeseq2_0[2],
            color='orchid', marker='o', s=50, label='Inicio 4')

ax1.scatter(C_Inicialeseq2_1[0], C_Inicialeseq2_1[1], C_Inicialeseq2_1[2],
            color='navy', marker='o', s=50, label='Inicio 5')

ax1.scatter(C_Inicialeseq2_2[0], C_Inicialeseq2_2[1], C_Inicialeseq2_2[2],
            color='darkmagenta', marker='o', s=50, label='Inicio 6')


# ------------------------------
# Puntos finales de las trayectorias
# ------------------------------

ax1.scatter(sol.y[0, -1], sol.y[1, -1], sol.y[2, -1],
            color='b', marker='s', s=70, label='Final 1')

ax1.scatter(sol2.y[0, -1], sol2.y[1, -1], sol2.y[2, -1],
            color='g', marker='s', s=70, label='Final 2')

ax1.scatter(sol3.y[0, -1], sol3.y[1, -1], sol3.y[2, -1],
            color='r', marker='s', s=70, label='Final 3')

ax1.scatter(sol4.y[0, -1], sol4.y[1, -1], sol4.y[2, -1],
            color='orchid', marker='s', s=70, label='Final 4')

ax1.scatter(sol5.y[0, -1], sol5.y[1, -1], sol5.y[2, -1],
            color='purple', marker='s', s=70, label='Final 5')

ax1.scatter(sol6.y[0, -1], sol6.y[1, -1], sol6.y[2, -1],
            color='darkmagenta', marker='s', s=70, label='Final 6')


# ------------------------------
# Puntos de equilibrio
# ------------------------------

ax1.scatter(Solucion_1.x[0], Solucion_1.x[1], Solucion_1.x[2],
            color='k', marker='X', s=150,
            label='Punto de Equilibrio 16.393 -769.676 2')

ax1.scatter(Solucion_2.x[0], Solucion_2.x[1], Solucion_2.x[2],
            color='r', marker='X', s=150,
            label='Punto de Equilibrio -15.713 835.676 2')


# ------------------------------
# Configuración del gráfico
# ------------------------------

ax1.text2D(
    0.5, -0.12,
    'Figura 3: Diagrama de fase 3D - Modelo no lineal',
    transform=ax1.transAxes,
    ha='center',
    va='top',
    fontsize=11,
    fontweight='bold'
)

ax1.set_xlabel('C (Concentración)')
ax1.set_ylabel('M (Masa acumulada)')
ax1.set_zlabel('S (Fuente)')
ax1.legend(fontsize=6)

# ------------------------------
# DERECHA: MODELO LINEALIZADO
# ------------------------------

ax2.plot(sol1_lin.y[0], sol1_lin.y[1], sol1_lin.y[2], color='b', label='Trayectoria 1')
ax2.plot(sol2_lin.y[0], sol2_lin.y[1], sol2_lin.y[2], color='g', label='Trayectoria 2')
ax2.plot(sol3_lin.y[0], sol3_lin.y[1], sol3_lin.y[2], color='r', label='Trayectoria 3')

# puntos iniciales
ax2.scatter(sol1_lin.y[0,0], sol1_lin.y[1,0], sol1_lin.y[2,0], color='b', marker='o', s=50)
ax2.scatter(sol2_lin.y[0,0], sol2_lin.y[1,0], sol2_lin.y[2,0], color='g', marker='o', s=50)
ax2.scatter(sol3_lin.y[0,0], sol3_lin.y[1,0], sol3_lin.y[2,0], color='r', marker='o', s=50)

# puntos finales
ax2.scatter(sol1_lin.y[0,-1], sol1_lin.y[1,-1], sol1_lin.y[2,-1], color='b', marker='s', s=60)
ax2.scatter(sol2_lin.y[0,-1], sol2_lin.y[1,-1], sol2_lin.y[2,-1], color='g', marker='s', s=60)
ax2.scatter(sol3_lin.y[0,-1], sol3_lin.y[1,-1], sol3_lin.y[2,-1], color='r', marker='s', s=60)

# equilibrio
ax2.scatter(X_eq[0], X_eq[1], X_eq[2],color='k', marker='X', s=150, label='Equilibrio')

ax2.text2D(
    0.5, -0.12,
    'Figura 4: Diagrama de fase 3D - Modelo linealizado al rededor del punto [16.393 -769.676  2]',
    transform=ax2.transAxes,
    ha='center',
    va='top',
    fontsize=11,
    fontweight='bold'
)

ax2.set_xlabel('C (Concentración)')
ax2.set_ylabel('M (Masa acumulada)')
ax2.set_zlabel('S (Fuente)')
ax2.legend(fontsize=8)

# misma vista en ambos
ax1.view_init(elev=20, azim=-60)
ax2.view_init(elev=20, azim=-60)

# ==========================================
# Polos y ceros de la función de transferencia
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
fig2.subplots_adjust(bottom=0.20)

# Título debajo de la figura
fig2.text(
    0.5, 0.04,
    'Figura 5: Mapa de polos y ceros',
    ha='center',
    va='center',
    fontsize=11,
    fontweight='bold'
)

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
axes[0].set_xlabel('Tiempo (min)')
axes[0].set_ylabel('Respuesta de C')
axes[0].grid(True, alpha=0.3)
# Respuesta al escalón
axes[1].plot(
    t_escalon, y_escalon,
    color='red',
    linewidth=2
)
axes[1].set_xlabel('Tiempo (min)')
axes[1].set_ylabel('Respuesta de C')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.subplots_adjust(bottom=0.14)

# Título debajo de la columna izquierda
fig.text(
    0.27, 0.035,
    'Figura 6: Respuesta al impulso',
    ha='center',
    va='center',
    fontsize=11,
    fontweight='bold'
)

# Título debajo de la columna derecha
fig.text(
    0.76, 0.035,
    'Figura 7: Respuesta al escalón',
    ha='center',
    va='center',
    fontsize=11,
    fontweight='bold'
)

plt.show()


