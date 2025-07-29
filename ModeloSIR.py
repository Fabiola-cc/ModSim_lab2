import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del modelo
N = 1000           # Población total
I0 = 10             # Infectados iniciales
R0 = 0             # Recuperados iniciales
S0 = N - I0 - R0   # Susceptibles iniciales

beta = 0.3         # Tasa de contagio
gamma = 0.1        # Tasa de recuperación
R_0 = 3 # Confirmamos que R₀ = 3

# Tiempo (en días)
t = np.linspace(0, 100, 100)

# Ecuaciones diferenciales del modelo SIR
def deriv(t, y, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Condiciones iniciales
y0 = S0, I0, R0

# Resolver el sistema
sol = solve_ivp(
    fun=lambda t, y: deriv(t, y, N, beta, gamma),
    t_span=(0, 100),
    y0=y0,
    t_eval=t
)

# Extraer resultados
t = sol.t
S, I, R = sol.y

# Graficar resultados
plt.figure(figsize=(10,6))
plt.plot(t, S, 'b', label='Susceptibles')
plt.plot(t, I, 'r', label='Infectados')
plt.plot(t, R, 'g', label='Recuperados')
plt.xlabel('Días')
plt.ylabel('Número de personas')
plt.title(f'Modelo SIR con R₀ = {R_0}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
