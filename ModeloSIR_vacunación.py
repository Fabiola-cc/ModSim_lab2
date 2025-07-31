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
herd_immunity_threshold = 1 - 1/R_0

# Función derivada del modelo
def deriv(t, y, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Paso 1: simular de 0 a 30 días
t_span1 = (0, 30)
t_eval1 = np.linspace(*t_span1, 31)
y0_1 = [S0, I0, R0]

sol1 = solve_ivp(
    fun=lambda t, y: deriv(t, y, N, beta, gamma),
    t_span=t_span1,
    y0=y0_1,
    t_eval=t_eval1
)

# Aplicar vacunación el día 30
S_vac = sol1.y[0, -1]
I_vac = sol1.y[1, -1]
R_vac = sol1.y[2, -1]

vacunados = 0.5 * S_vac
S2 = S_vac - vacunados
R2 = R_vac + vacunados
y0_2 = [S2, I_vac, R2]

# Paso 2: continuar de día 30 a 100
t_span2 = (30, 100)
t_eval2 = np.linspace(*t_span2, 70)

sol2 = solve_ivp(
    fun=lambda t, y: deriv(t, y, N, beta, gamma),
    t_span=t_span2,
    y0=y0_2,
    t_eval=t_eval2
)

# Unir resultados
t = np.concatenate((sol1.t, sol2.t))
S = np.concatenate((sol1.y[0], sol2.y[0]))
I = np.concatenate((sol1.y[1], sol2.y[1]))
R = np.concatenate((sol1.y[2], sol2.y[2]))

# Punto máximo de infección
max_I = np.max(I)
t_max_I = t[np.argmax(I)]

# Umbral de inmunidad
herd_S_threshold = N * (1 - herd_immunity_threshold)
herd_idx = np.where(S < herd_S_threshold)[0]
t_herd = t[herd_idx[0]] if herd_idx.size > 0 else None
R_herd = R[herd_idx[0]] if herd_idx.size > 0 else None

# Graficar
plt.figure(figsize=(10,6))
plt.plot(t, S, 'b', label='Susceptibles')
plt.plot(t, I, 'r', label='Infectados')
plt.plot(t, R, 'g', label='Recuperados')

# Anotar pico de infección
plt.plot(t_max_I, max_I, 'ro')
plt.annotate(f'Pico infección\n{int(max_I)} infectados\ndía {int(t_max_I)}',
             xy=(t_max_I, max_I), xytext=(t_max_I + 5, max_I + 20),
             arrowprops=dict(arrowstyle='->'), fontsize=9)

# Anotar umbral de inmunidad
if t_herd:
    plt.axvline(t_herd, color='gray', linestyle='--', label='Umbral inmunidad')
    plt.annotate(f'Umbral inmunidad\ndía {int(t_herd)}\nR ≈ {int(R_herd)}',
                 xy=(t_herd, R_herd), xytext=(t_herd - 25, R_herd + 50),
                 arrowprops=dict(arrowstyle='->'), fontsize=9)

# Marcar vacunación
plt.axvline(30, color='purple', linestyle=':', label='Vacunación 50% de S')
plt.annotate('Vacunación masiva\n(50% de S)', 
             xy=(30, I[np.where(t == 30)][0]), 
             xytext=(32, 10), 
             fontsize=9, color='purple')

# Estética
plt.xlabel('Días')
plt.ylabel('Personas')
plt.title(f'Modelo SIR con vacunación (R₀ = {R_0:.1f})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
