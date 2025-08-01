import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema
dt = 1.0  # Paso de tiempo (Δt) = 1 día
tiempo_total = 50  # días
retardo = 5  # días (τ)

# Condiciones iniciales
stock_inicial = 0  # unidades
entrada_inicial = 10  # unidades/día
entrada_cambio = 20  # unidades/día (después del día 25)
dia_cambio = 25

# Arrays para almacenar resultados
tiempo = np.arange(0, tiempo_total + dt, dt)
stock = np.zeros(len(tiempo))
salida = np.zeros(len(tiempo))
entrada = np.zeros(len(tiempo))

# Condición inicial
stock[0] = stock_inicial

# Simulación usando integración de Euler
for i in range(len(tiempo)):
    # Definir entrada (cambio repentino en día 25)
    if tiempo[i] < dia_cambio:
        entrada[i] = entrada_inicial
    else:
        entrada[i] = entrada_cambio
    
    # Calcular salida: stock/retardo
    salida[i] = stock[i] / retardo
    
    # Integración de Euler para el siguiente paso
    if i < len(tiempo) - 1:
        # dS/dt = entrada - salida
        dstock_dt = entrada[i] - salida[i]
        stock[i + 1] = stock[i] + dstock_dt * dt

# Calcular equilibrios teóricos
equilibrio_inicial = entrada_inicial * retardo  # 10 * 5 = 50
equilibrio_final = entrada_cambio * retardo     # 20 * 5 = 100

# Crear la gráfica
plt.figure(figsize=(12, 8))

# Gráfica principal
plt.subplot(2, 1, 1)
plt.plot(tiempo, stock, 'b-', linewidth=3, label='Stock')
plt.plot(tiempo, salida, 'r-', linewidth=2, label='Flujo de Salida')
plt.plot(tiempo, entrada, 'g--', linewidth=2, label='Entrada')

# Líneas de equilibrio
plt.axhline(y=equilibrio_inicial, color='orange', linestyle=':', linewidth=2, 
           label=f'Equilibrio inicial ({equilibrio_inicial})')
plt.axhline(y=equilibrio_final, color='purple', linestyle=':', linewidth=2, 
           label=f'Equilibrio final ({equilibrio_final})')
plt.axvline(x=dia_cambio, color='gray', linestyle='--', alpha=0.7, 
           label=f'Cambio entrada (día {dia_cambio})')

plt.xlabel('Tiempo (días)')
plt.ylabel('Unidades')
plt.title('Sistema de Retardo de Primer Orden\n(Stock inicial=0, Entrada=10→20, τ=5 días)')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica de análisis del equilibrio
plt.subplot(2, 1, 2)
# Mostrar la diferencia entre stock y equilibrio teórico
equilibrio_teorico = np.where(tiempo < dia_cambio, equilibrio_inicial, equilibrio_final)
diferencia = stock - equilibrio_teorico

plt.plot(tiempo, diferencia, 'm-', linewidth=2, label='Stock - Equilibrio Teórico')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.axvline(x=dia_cambio, color='gray', linestyle='--', alpha=0.7)

plt.xlabel('Tiempo (días)')
plt.ylabel('Diferencia (unidades)')
plt.title('Convergencia al Equilibrio')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Imprimir resultados clave
print("=== ANÁLISIS DEL SISTEMA DE RETARDO DE PRIMER ORDEN ===")
print(f"Parámetros: τ = {retardo} días, Δt = {dt} día")
print(f"Entrada inicial: {entrada_inicial} unidades/día")
print(f"Entrada final: {entrada_cambio} unidades/día (desde día {dia_cambio})")
print(f"Stock inicial: {stock_inicial} unidades")
print()

print("EQUILIBRIOS TEÓRICOS:")
print(f"Equilibrio inicial = Entrada × τ = {entrada_inicial} × {retardo} = {equilibrio_inicial} unidades")
print(f"Equilibrio final = Entrada × τ = {entrada_cambio} × {retardo} = {equilibrio_final} unidades")
print()

print("RESULTADOS DE LA SIMULACIÓN:")
print(f"Stock al día 24 (antes del cambio): {stock[24]:.2f} unidades")
print(f"Stock al día 25 (día del cambio): {stock[25]:.2f} unidades")
print(f"Stock al día 50 (final): {stock[50]:.2f} unidades")
print(f"Salida al día 50: {salida[50]:.2f} unidades/día")
print()

print("ANÁLISIS DE CONVERGENCIA:")
print(f"Tiempo para 63.2% del cambio inicial: ~{retardo} días")
print(f"Tiempo para 95% del cambio inicial: ~{3*retardo} días")
print(f"Diferencia final (Stock - Equilibrio): {stock[50] - equilibrio_final:.2f} unidades")

# Verificación de equilibrio (Entrada = Salida)
print()
print("VERIFICACIÓN DE EQUILIBRIO (Entrada = Salida):")
print(f"Entrada final: {entrada[50]:.2f} unidades/día")
print(f"Salida final: {salida[50]:.2f} unidades/día")
print(f"Diferencia: {abs(entrada[50] - salida[50]):.3f} unidades/día")

# Análisis del comportamiento exponencial
print()
print("COMPORTAMIENTO EXPONENCIAL:")
tiempo_constante = tiempo[np.where(tiempo <= 25)]  # Antes del cambio
stock_constante = stock[np.where(tiempo <= 25)]
teorico_constante = equilibrio_inicial * (1 - np.exp(-tiempo_constante/retardo))

error_max = np.max(np.abs(stock_constante - teorico_constante))
print(f"Error máximo vs. solución analítica (primera fase): {error_max:.3f} unidades")