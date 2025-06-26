import numpy as np

class Pendulum:
    """
    Clase para simular un péndulo simple.
    """
    # Valores predeterminados como atributos de clase para acceso externo (ej. desde main.py)
    DEFAULT_LENGTH = 1.0  # Longitud de la cuerda en metros
    DEFAULT_MASS = 1.0  # Masa de la lenteja en kg (no afecta el período en el modelo simple)
    DEFAULT_GRAVITY = 9.81  # Aceleración debida a la gravedad en m/s^2
    DEFAULT_THETA0 = np.pi / 4  # Ángulo inicial en radianes (45 grados)
    DEFAULT_OMEGA0 = 0.0  # Velocidad angular inicial en radianes/s

    def __init__(self, length=None, mass=None, gravity=None,
                 theta0=None, omega0=None):

        self.length = length if length is not None else Pendulum.DEFAULT_LENGTH
        self.mass = mass if mass is not None else Pendulum.DEFAULT_MASS
        self.gravity = gravity if gravity is not None else Pendulum.DEFAULT_GRAVITY
        self.theta = theta0 if theta0 is not None else Pendulum.DEFAULT_THETA0
        self.omega = omega0 if omega0 is not None else Pendulum.DEFAULT_OMEGA0

        self.time_elapsed = 0.0

        # Para almacenar el historial de la simulación
        self.history_time = [self.time_elapsed]
        self.history_theta = [self.theta]
        self.history_omega = [self.omega]

    def get_acceleration(self, theta_current):
        """
        Calcula la aceleración angular del péndulo.
        θ'' = -(g/L) * sin(θ)
        """
        if self.length == 0:
            return 0 # Evitar división por cero
        return -(self.gravity / self.length) * np.sin(theta_current)

    def step(self, dt):
        """
        Avanza la simulación un paso de tiempo 'dt' usando el método de Euler.
        Una mejora sería usar Runge-Kutta de cuarto orden (RK4) para mayor precisión.
        Por simplicidad, empezamos con Euler.
        """
        # Método de Euler explícito
        # acceleration = self.get_acceleration(self.theta)
        # self.omega += acceleration * dt
        # self.theta += self.omega * dt
        # self.time_elapsed += dt

        # Método de Euler-Cromer (más estable para osciladores)
        acceleration = self.get_acceleration(self.theta)
        self.omega += acceleration * dt
        self.theta += self.omega * dt # Usa la nueva omega para actualizar theta
        self.time_elapsed += dt

        # Guardar el estado actual en el historial
        self.history_time.append(self.time_elapsed)
        self.history_theta.append(self.theta)
        self.history_omega.append(self.omega)

    def simulate(self, t_max, dt):
        """
        Ejecuta la simulación hasta un tiempo máximo t_max con pasos de tiempo dt.
        """
        # Reiniciar historial si se llama a simulate múltiples veces
        self.history_time = [self.time_elapsed]
        self.history_theta = [self.theta]
        self.history_omega = [self.omega]

        current_time = self.time_elapsed
        while current_time < t_max:
            # Asegurar que el último paso no exceda t_max
            time_step = min(dt, t_max - current_time)
            self.step(time_step)
            current_time += time_step
            if current_time >= t_max: # Pequeña corrección por errores de flotantes
                self.time_elapsed = t_max


if __name__ == '__main__':
    # Ejemplo de uso (para pruebas rápidas)
    pendulum_sim = Pendulum(theta0=np.pi/3) # 60 grados

    time_simulation = 10.0  # segundos
    delta_t = 0.01         # paso de tiempo en segundos

    print(f"Simulando péndulo por {time_simulation} segundos con dt = {delta_t}s...")
    pendulum_sim.simulate(time_simulation, delta_t)
    print("Simulación completada.")

    # Imprimir algunos valores para verificar
    print(f"Tiempo final: {pendulum_sim.time_elapsed:.2f}s")
    print(f"Ángulo final: {np.rad2deg(pendulum_sim.theta):.2f} grados")
    print(f"Velocidad angular final: {np.rad2deg(pendulum_sim.omega):.2f} grad/s")
    print(f"Total de pasos almacenados: {len(pendulum_sim.history_time)}")

    # Podríamos graficar aquí con matplotlib si no estuviéramos en un script simple
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(pendulum_sim.history_time, np.rad2deg(pendulum_sim.history_theta))
    # plt.title('Posición Angular vs. Tiempo')
    # plt.xlabel('Tiempo (s)')
    # plt.ylabel('Ángulo (grados)')
    # plt.grid(True)

    # plt.subplot(2, 1, 2)
    # plt.plot(pendulum_sim.history_time, np.rad2deg(pendulum_sim.history_omega))
    # plt.title('Velocidad Angular vs. Tiempo')
    # plt.xlabel('Tiempo (s)')
    # plt.ylabel('Velocidad Angular (grados/s)')
    # plt.grid(True)

    import matplotlib.pyplot as plt # Asegurarse de que plt esté disponible
    plt.tight_layout()
    plt.show()

# Las funciones de animación y configuración de gráficos se quedan aquí por ahora,
# pero main.py las llama. Podrían moverse a un módulo de visualización separado
# si el proyecto creciera.
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def get_cartesian_coords(theta, length):
    """Convierte coordenadas polares (ángulo) a cartesianas para el péndulo."""
    x = length * np.sin(theta)
    y = -length * np.cos(theta) # Negativo porque el origen (0,0) es el punto de suspensión
    return x, y

def animate_pendulum(i, pendulum_obj, line, point, time_template, time_text, trace_line, trace_max_points=200):
    """
    Función de actualización para la animación.
    'i' es el número de frame.
    """
    # Obtenemos los datos del péndulo en el tiempo correspondiente al frame i
    # Asumimos que los pasos de tiempo de la simulación son consistentes
    # y podemos mapear 'i' a un índice en history_theta.
    # Esto es una simplificación; idealmente, la simulación avanza con la animación.
    # Para este ejemplo, asumimos que la simulación ya ocurrió.

    idx = i % len(pendulum_obj.history_theta) # Cicla la animación si los frames superan los datos
    current_theta = pendulum_obj.history_theta[idx]
    current_time = pendulum_obj.history_time[idx]

    x, y = get_cartesian_coords(current_theta, pendulum_obj.length)

    line.set_data([0, x], [0, y])
    point.set_data([x], [y])
    time_text.set_text(time_template % current_time)

    # Actualizar la estela (trace)
    if idx == 0: # Reiniciar la estela si la animación cicla
        trace_line.set_data([], [])

    trace_x, trace_y = trace_line.get_data()

    new_trace_x = list(trace_x)
    new_trace_y = list(trace_y)

    new_trace_x.append(x)
    new_trace_y.append(y)

    if len(new_trace_x) > trace_max_points:
        new_trace_x = new_trace_x[-trace_max_points:]
        new_trace_y = new_trace_y[-trace_max_points:]

    trace_line.set_data(new_trace_x, new_trace_y)

    return line, point, time_text, trace_line


def setup_animation(pendulum_obj, time_simulation, dt_simulation):
    """
    Configura y muestra la animación del péndulo.
    """
    # Ejecutar la simulación primero
    pendulum_obj.simulate(time_simulation, dt_simulation)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, autoscale_on=False)

    # Configuración de los límites del eje
    # El péndulo se mueve entre -length y +length en x, y 0 y -length en y (respecto al punto de suspensión)
    # Damos un poco de margen
    margin = 0.2 * pendulum_obj.length
    ax.set_xlim(-pendulum_obj.length - margin, pendulum_obj.length + margin)
    ax.set_ylim(-pendulum_obj.length - margin, margin) # y va hacia abajo

    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Simulación de Péndulo Simple')
    ax.set_xlabel('Posición X (m)')
    ax.set_ylabel('Posición Y (m)')

    # Elementos de la animación
    line, = ax.plot([], [], 'o-', lw=2, color='blue', markersize=3) # Cuerda y punto de suspensión
    point, = ax.plot([], [], 'o', lw=2, color='red', markersize=10) # Masa del péndulo
    trace_line, = ax.plot([], [], ':', lw=1, color='gray', alpha=0.7) # Estela

    time_template = 'Tiempo = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    # Número de frames: basado en la cantidad de datos simulados
    num_frames = len(pendulum_obj.history_time)

    # Intervalo entre frames en milisegundos.
    # Queremos que la animación dure aproximadamente el tiempo real de simulación.
    # Si dt_simulation es muy pequeño, la animación será lenta. Si es grande, será rápida.
    # Un intervalo de 20 ms corresponde a 50 FPS.
    # Si queremos que la animación dure T_sim segundos reales, y tenemos N_frames,
    # el intervalo = (T_sim * 1000) / N_frames.
    # O, más simple, si cada frame representa dt_simulation segundos, el intervalo es dt_simulation * 1000.
    # Sin embargo, la velocidad de la animación también depende de cuán rápido Matplotlib pueda dibujar.
    # Usaremos un valor fijo y ajustaremos si es necesario, o basarlo en dt_simulation.

    # Si dt_simulation = 0.01s, interval = 10ms
    # Si dt_simulation = 0.02s, interval = 20ms
    # Para una visualización fluida, un intervalo de 20-50 ms suele ser bueno.
    # Si la simulación tiene muchos pasos, podemos submuestrear para la animación.
    # Por ahora, usemos todos los puntos simulados.

    # El parámetro 'interval' en FuncAnimation es el retardo entre frames en milisegundos.
    # Si dt_simulation es el paso de tiempo de la simulación, entonces interval = dt_simulation * 1000
    # para que la animación intente correr en tiempo real.
    # Sin embargo, si dt_simulation es muy pequeño (ej. 0.001s), interval sería 1ms, lo cual es
    # demasiado rápido para que matplotlib lo maneje y la animación se verá más lenta que en tiempo real.
    # Un valor común para 'interval' es 20 (50 FPS).
    # Vamos a intentar que cada frame de animación corresponda a un paso de simulación.
    interval_ms = dt_simulation * 1000
    # Aseguramos un mínimo para que no sea demasiado rápido para renderizar
    interval_ms = max(20, interval_ms)


    ani = animation.FuncAnimation(fig, animate_pendulum, frames=num_frames,
                                  fargs=(pendulum_obj, line, point, time_template, time_text, trace_line),
                                  interval=interval_ms, blit=True, repeat=True)
                                  # repeat=False para que no se repita

    plt.show()

if __name__ == '__main__':
    # Ejemplo de uso (para pruebas rápidas)
    pendulum_sim = Pendulum(length=1.5, theta0=np.pi/2, omega0=0.1) # 90 grados iniciales

    time_simulation = 10.0  # segundos
    delta_t_sim = 0.02      # paso de tiempo de simulación en segundos

    # No llamamos a pendulum_sim.simulate() aquí, setup_animation lo hará.
    print(f"Configurando animación para {time_simulation}s con dt_sim = {delta_t_sim}s...")
    setup_animation(pendulum_sim, time_simulation, delta_t_sim)

    # El código anterior de impresión y gráficos se ejecutaría después de cerrar la ventana de animación.
    print("Animación cerrada.")
    print(f"Tiempo final de simulación: {pendulum_sim.time_elapsed:.2f}s")
    print(f"Ángulo final: {np.rad2deg(pendulum_sim.theta):.2f} grados")
    print(f"Velocidad angular final: {np.rad2deg(pendulum_sim.omega):.2f} grad/s")
    print(f"Total de pasos almacenados: {len(pendulum_sim.history_time)}")

    # Graficar después de la animación
    # import matplotlib.pyplot as plt # Ya importado arriba
    plt.figure(figsize=(12, 8)) # Crear una nueva figura para los gráficos

    plt.subplot(2, 1, 1)
    plt.plot(pendulum_sim.history_time, np.rad2deg(np.array(pendulum_sim.history_theta)), label='Ángulo (θ)')
    plt.title('Posición Angular vs. Tiempo (Después de Animación)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Ángulo (grados)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(pendulum_sim.history_time, np.rad2deg(np.array(pendulum_sim.history_omega)), label='Velocidad Angular (ω)', color='orange')
    plt.title('Velocidad Angular vs. Tiempo (Después de Animación)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad Angular (grados/s)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout(pad=3.0) # Añadir un poco de padding para que los títulos no se solapen
    # plt.show() # main.py se encargará de llamar a plt.show() para los gráficos finales
