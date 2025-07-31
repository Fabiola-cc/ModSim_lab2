import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del modelo
N = 1000
I0 = 10
R0 = 0
S0 = N - I0 - R0

beta = 0.3
gamma = 0.1
R_0 = beta / gamma
herd_immunity_threshold = 1 - 1/R_0  # 66.7% para R0 = 3

# Función derivada
def deriv(t, y, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Condiciones iniciales
y0 = [S0, I0, R0]
t_eval = np.linspace(0, 100, 100)

# Resolver
sol = solve_ivp(
    fun=lambda t, y: deriv(t, y, N, beta, gamma),
    t_span=(0, 100),
    y0=y0,
    t_eval=t_eval
)
t = sol.t
S, I, R = sol.y

# 1. Encontrar el punto máximo de infección
max_I = np.max(I)
t_max_I = t[np.argmax(I)]

# 2. Calcular cuándo se alcanza el umbral de inmunidad
herd_S_threshold = N * (1 - herd_immunity_threshold)  # S = 33.3% de N
herd_idx = np.where(S < herd_S_threshold)[0]
t_herd = t[herd_idx[0]] if herd_idx.size > 0 else None
R_herd = R[herd_idx[0]] if herd_idx.size > 0 else None

# Graficar
plt.figure(figsize=(10,6))
plt.plot(t, S, 'b', label='Susceptibles')
plt.plot(t, I, 'r', label='Infectados')
plt.plot(t, R, 'g', label='Recuperados')

# Anotar el pico de infección
plt.plot(t_max_I, max_I, 'ro')
plt.annotate(f'Pico de infección\n{int(max_I)} infectados\ndía {int(t_max_I)}',
             xy=(t_max_I, max_I), xytext=(t_max_I + 5, max_I + 20),
             arrowprops=dict(arrowstyle='->'), fontsize=9)

# Anotar el umbral de inmunidad de grupo
if t_herd:
    plt.axvline(t_herd, color='gray', linestyle='--', label='Umbral inmunidad')
    plt.annotate(f'Umbral de inmunidad\ndía {int(t_herd)}\nR ≈ {int(R_herd)}',
                 xy=(t_herd, R_herd), xytext=(t_herd + 5, R_herd + 50),
                 arrowprops=dict(arrowstyle='->'), fontsize=9)

# Estética general
plt.xlabel('Días')
plt.ylabel('Personas')
plt.title(f'Modelo SIR con solve_ivp (R₀ = {R_0:.1f})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()