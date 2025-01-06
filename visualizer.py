import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from environment import LightTrackingEnv  # Deine Umgebung

# Initialisiere die Umgebung
env = LightTrackingEnv()

# Funktion, um die Umgebungsschritte zu simulieren
def update(frame, servo_scatter, light_scatter):
    # Simuliere einen Schritt
    action = np.random.choice(env.action_space.n)  # Zufällige Aktion (später durch Modell ersetzen)
    obs, reward, done, _, _ = env.step(action)

    # Aktualisiere die Positionen
    servo_position = env.servo_position
    light_source = env.light_source

    # Aktualisiere die Scatter-Daten
    servo_scatter.set_offsets([servo_position])
    light_scatter.set_offsets([light_source * 180])  # Lichtquelle auf Skala 0-180 transformieren

    # Titel mit Informationen
    plt.title(f"Reward: {reward:.2f} | Servo: {servo_position} | Light: {light_source}")
    return servo_scatter, light_scatter

# Erstelle das Plot-Layout
fig, ax = plt.subplots()
ax.set_xlim(0, 180)
ax.set_ylim(0, 180)
ax.set_xlabel("X-Position")
ax.set_ylabel("Y-Position")
ax.grid()

# Scatter-Objekte für Servo und Lichtquelle
servo_scatter = ax.scatter([], [], c="blue", label="Servo Position")
light_scatter = ax.scatter([], [], c="orange", label="Light Source")
ax.legend()

# Animation
ani = animation.FuncAnimation(fig, update, frames=200, fargs=(servo_scatter, light_scatter), interval=100, blit=False)

# Zeige die Animation
plt.show()
