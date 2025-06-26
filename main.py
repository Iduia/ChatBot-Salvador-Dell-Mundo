import numpy as np
from pendulum import Pendulum, setup_animation
import matplotlib.pyplot as plt

def get_float_input(prompt, default_value):
    """Solicita al usuario un valor flotante con un valor predeterminado."""
    while True:
        try:
            user_input = input(f"{prompt} (predeterminado: {default_value}): ")
            if not user_input:
                return default_value
            return float(user_input)
        except ValueError:
            print("Entrada no válida. Por favor, ingrese un número.")

def get_angle_input(prompt, default_value_deg):
    """Solicita al usuario un ángulo en grados y lo convierte a radianes."""
    while True:
        try:
            user_input = input(f"{prompt} en grados (predeterminado: {default_value_deg}): ")
            if not user_input:
                return np.deg2rad(default_value_deg)
            return np.deg2rad(float(user_input))
        except ValueError:
            print("Entrada no válida. Por favor, ingrese un número para el ángulo.")

def main():
    print("=== Simulación de Péndulo Simple ===")

    # Solicitar parámetros al usuario
    length = get_float_input("Longitud de la cuerda (L) en metros", Pendulum.DEFAULT_LENGTH)
    mass = get_float_input("Masa de la lenteja (m) en kg", Pendulum.DEFAULT_MASS)
    gravity = get_float_input("Aceleración de la gravedad (g) en m/s^2", Pendulum.DEFAULT_GRAVITY)
    theta0_deg = np.rad2deg(Pendulum.DEFAULT_THETA0) # Convertir default a grados para el prompt
    theta0 = get_angle_input("Ángulo inicial (θ₀)", theta0_deg)
    omega0_deg_s = np.rad2deg(Pendulum.DEFAULT_OMEGA0) # Convertir default a grados/s para el prompt
    omega0_rad_s = get_angle_input("Velocidad angular inicial (ω₀) en grados/s", omega0_deg_s)

    print("\n--- Parámetros de Simulación ---")
    time_simulation = get_float_input("Duración de la simulación (t_max) en segundos", 10.0)
    dt_simulation = get_float_input("Paso de tiempo para la simulación (dt) en segundos", 0.02)

    # Crear instancia del péndulo con los parámetros del usuario
    pendulum_sim = Pendulum(length=length, mass=mass, gravity=gravity,
                            theta0=theta0, omega0=omega0_rad_s)

    # Configurar y mostrar la animación
    # La simulación se ejecuta dentro de setup_animation
    print(f"\nConfigurando animación para {time_simulation}s con dt_sim = {dt_simulation}s...")
    # Nota: setup_animation en pendulum.py llama a plt.show() para la animación.
    # Esto bloqueará la ejecución hasta que se cierre la ventana de animación.
    setup_animation(pendulum_sim, time_simulation, dt_simulation)

    print("Animación cerrada.")

    # Mostrar gráficos de posición y velocidad angular
    print("Generando gráficos...")
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(pendulum_sim.history_time, np.rad2deg(np.array(pendulum_sim.history_theta)), label='Ángulo (θ)')
    plt.title('Posición Angular vs. Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Ángulo (grados)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(pendulum_sim.history_time, np.rad2deg(np.array(pendulum_sim.history_omega)), label='Velocidad Angular (ω)', color='orange')
    plt.title('Velocidad Angular vs. Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad Angular (grados/s)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout(pad=3.0)
    # plt.show() se llama aquí para mostrar los gráficos después de la animación.
    # Si setup_animation tiene su propio show(), este podría abrir una nueva ventana vacía si no hay nada más que mostrar.
    # Es mejor que setup_animation no llame a show() y que el control de mostrar figuras
    # se maneje en main.py.
    plt.show() # Muestra los gráficos de ángulo y velocidad. La animación ya tuvo su propio plt.show().

    print("\nSimulación y visualización completadas.")

if __name__ == '__main__':
    main()
