import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from utils import calculate_reward
import torch

class LightTrackingEnv(gym.Env):
    def __init__(self):
        super(LightTrackingEnv, self).__init__()
        self.light_intensity = 10.0  # Beispielwert
        self.sensor_sensitivity = 5.0  # Empfindlichkeit des LDR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Standard-Radien
        self.light_radius = self.light_intensity * 5.0  # Beispiel: Lichtintensität beeinflusst den Radius
        self.sensor_radius = [
            self.sensor_sensitivity * 4.0,  # Radius für LDR 1
            self.sensor_sensitivity * 4.0,  # Radius für LDR 2
            self.sensor_sensitivity * 4.0,  # Radius für LDR 3
            self.sensor_sensitivity * 4.0   # Radius für LDR 4
        ]

        self.action_space = spaces.Discrete(5)  # 5 Aktionen: Links, Rechts, Hoch, Runter, Stopp
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)  # LDR-Werte
        self.servo_position = [90, 90]  # Anfangswinkel horizontal, vertikal
        self.light_source = np.array([0.5, 0.5])  # Initiale Position der Lichtquelle
        self.previous_action = 4  # Start mit "Stopp"
        self.current_action = 4
        self.ldr_values_log = []
        self.servo_positions_log = []
        self.rewards_log = []
        self.light_source_log = []

        # Lokale Definition der Parameter
        self.step_size = 5
        self.ldr_positions = np.array([
            [0, 1],   # Oben rechts
            [-1, 0],  # Oben links
            [0, -1],  # Unten links
            [1, 0]    # Unten rechts
        ])

    def seed(self, seed=None):
        """
        Setzt den Zufallszahlengenerator.
        """
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        random.seed(seed)
    
    def reset(self, seed=None, options=None):
        self.servo_position = [random.randint(30, 150), random.randint(30, 150)]
        self.light_source = np.random.uniform(0.2, 0.8, size=2)  # Breiterer Bereich
        self.previous_action = 4  # Stopp
        self.current_action = 4
        self.ldr_values_log = []
        self.servo_positions_log = []
        self.rewards_log = []
        self.light_source_log = []
        return self._calculate_ldr_values(), {}
    
    def _is_out_of_bounds(self):
        servo_position_array = np.array(self.servo_position)  # Konvertiere in NumPy-Array
        return not np.all((0 <= servo_position_array) & (servo_position_array <= 180))
     
    def step(self, action):
        # Speichere die aktuelle Aktion
        self.current_action = action

        # Adaptiver Schritt
        distance_to_light = np.linalg.norm(self.light_source * 180 - self.servo_position)
        adaptive_step = max(3, int(self.step_size * (1 - distance_to_light / 180)))

        if action == 0:  # Links
            self.servo_position[0] = max(0, self.servo_position[0] - adaptive_step)
        elif action == 1:  # Rechts
            self.servo_position[0] = min(180, self.servo_position[0] + adaptive_step)
        elif action == 2:  # Hoch
            self.servo_position[1] = max(0, self.servo_position[1] - adaptive_step)
        elif action == 3:  # Runter
            self.servo_position[1] = min(180, self.servo_position[1] + adaptive_step)

        self.ldr_positions = np.array([
            [self.servo_position[0], self.servo_position[1] + self.sensor_radius[0]],  # Oben
            [self.servo_position[0] - self.sensor_radius[1], self.servo_position[1]],  # Links
            [self.servo_position[0], self.servo_position[1] - self.sensor_radius[2]],  # Unten
            [self.servo_position[0] + self.sensor_radius[3], self.servo_position[1]]   # Rechts
        ])

        # Berechne LDR-Werte
        ldr_values = self._calculate_ldr_values()

        # Neue Belohnungsberechnung
        reward, ldr_reward, balance_penalty, position_penalty, raw_reward = calculate_reward(ldr_values, self.servo_position, self.light_source)
        #reward, intensity_bonus, spread_penalty = calculate_reward(ldr_values)
        reward = max(reward, 0)  # Reward bleibt mindestens bei 0

        # Speichere die aktuelle Aktion als vorherige Aktion
        self.previous_action = action

        # In der Schritt-Funktion
        self.ldr_values_log.append(ldr_values)
        self.servo_positions_log.append(list(self.servo_position))
        self.rewards_log.append(reward)
        self.light_source_log.append(self.light_source)

        done = reward > 3.5 or self._is_out_of_bounds()

        self._move_light_source()

        return ldr_values, reward, done, False, {}
    
    def _calculate_ldr_values(self):
        # Berechnung der Distanzen
        distances = np.linalg.norm(self.light_source - self.ldr_positions, axis=1)

        # Lichtintensitätsberechnung
        light_intensity = np.array([
            1 if distance <= self.light_radius else 1 / (1 + (distance - self.light_radius) ** 2)
            for distance in distances
        ])

        sensor_effect = np.array([
            1.2 if distance <= ldr_radius else 1
            for distance, ldr_radius in zip(distances, self.sensor_radius)
        ])

        # Finaler Messwert
        final_values = light_intensity * sensor_effect
        return final_values / final_values.max() if final_values.max() > 0 else final_values

    def _move_light_source(self, move_range=0.005, stationary=False):
        if not stationary:
            direction = np.random.choice([-1, 1], size=2)
            self.light_source += direction * np.random.uniform(0, move_range, size=2)
            self.light_source = np.clip(self.light_source, 0, 1)

    def save_logs(self, filename="logs.npz"):
        """
        Speichert die Logs als NumPy-Zip-Datei.
        """
        np.savez(
            filename,
            ldr_values=np.array(self.ldr_values_log),
            servo_positions=np.array(self.servo_positions_log),
            rewards=np.array(self.rewards_log),
            light_source_positions=np.array(self.light_source_log),  # Lichtquellenpositionen hinzufügen
        )
        print(f"Logs wurden unter '{filename}' gespeichert.")

    def _calculate_sensor_values(self):
        sensor_radius = 5  # Beispiel: Sensor hat einen Radius von 5 Einheiten
        distances = np.linalg.norm(self.light_source - self.ldr_positions, axis=1)  # Distanz zu LDRs

        # Innerhalb des Sensor-Radius messen wir intensiver (Bonus)
        sensor_effect = np.array([
            1.2 if distance <= ldr_radius else 1
            for distance, ldr_radius in zip(distances, self.sensor_radius)
        ])

        return sensor_effect