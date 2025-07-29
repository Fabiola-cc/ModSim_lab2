import numpy as np
import matplotlib.pyplot as plt

# Parámetros
dias = 200
objetivo = 50
retraso_total = 15       # retraso medio total
orden_erlang = 3         # número de etapas en el retraso Erlang
dt = 1.0                 # tamaño del paso (1 día)
tiempo = np.arange(0, dias, dt)

# Inicialización
stock_real = np.zeros_like(tiempo)
flujo = np.zeros_like(tiempo)
delay_stages = [0.0 for _ in range(orden_erlang)]  # etapas del retraso

# Constante de tiempo de cada etapa
tau = retraso_total / orden_erlang

# Simulación
for t in range(1, len(tiempo)):
    # El percibido es la salida de la última etapa
    stock_percibido = delay_stages[-1]

    # Flujo de control (retroalimentación con el stock percibido)
    flujo[t] = (objetivo - stock_percibido) / retraso_total

    # Actualizar el stock real
    stock_real[t] = stock_real[t - 1] + flujo[t] * dt

    # Propagar el stock real a través de las etapas del retraso (Erlang como cadena de tanques)
    entrada = stock_real[t]
    nueva_etapa = [0.0 for _ in range(orden_erlang)]
    for i in range(orden_erlang):
        if i == 0:
            nueva_etapa[i] = delay_stages[i] + (entrada - delay_stages[i]) * dt / tau
        else:
            nueva_etapa[i] = delay_stages[i] + (delay_stages[i - 1] - delay_stages[i]) * dt / tau
    delay_stages = nueva_etapa

# ----------------------------
# Visualización
# ----------------------------

stock_percibido_total = [0.0] * len(tiempo)
delay_stages = [0.0 for _ in range(orden_erlang)]
for t in range(1, len(tiempo)):
    entrada = stock_real[t]
    nueva_etapa = [0.0 for _ in range(orden_erlang)]
    for i in range(orden_erlang):
        if i == 0:
            nueva_etapa[i] = delay_stages[i] + (entrada - delay_stages[i]) * dt / tau
        else:
            nueva_etapa[i] = delay_stages[i] + (delay_stages[i - 1] - delay_stages[i]) * dt / tau
    delay_stages = nueva_etapa
    stock_percibido_total[t] = delay_stages[-1]

# Gráfico
plt.figure(figsize=(12, 6))
plt.plot(tiempo, stock_real, label="Stock real", linewidth=2)
plt.plot(tiempo, stock_percibido_total, label="Stock percibido (con retraso)", linestyle='--')

# Resaltar sobrecorrecciones
cruces = np.where(np.diff(np.sign(stock_real - objetivo)))[0]
for c in cruces:
    plt.axvspan(tiempo[c], tiempo[c+1], color='orange', alpha=0.2, label='Sobrecorrección' if c == cruces[0] else "")

plt.axhline(y=objetivo, color='gray', linestyle=':', label='Objetivo')
plt.xlabel("Días")
plt.ylabel("Unidades")
plt.title("Sistema con Retraso de Percepción (Modelo Erlang)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
