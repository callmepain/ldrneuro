import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from utils import calculate_reward

class LightTrackingEnv(gym.Env):
    def __init__(self):
        super(LightTrackingEnv, self).__init__()
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

    def reset_alt(self, seed=None, options=None):
        self.servo_position = [random.randint(45, 135), random.randint(45, 135)]  # Zufällige Startposition
        self.light_source = np.random.uniform(0.3, 0.7, size=2)  # Lichtquelle zufällig innerhalb eines Bereichs
        self.previous_action = random.choice([0, 1, 2, 3])  # Zufällige Startaktion
        self.current_action = self.previous_action
        return self._calculate_ldr_values(), {}
    
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

    def step_alt(self, action):
        # Speichere die aktuelle Aktion
        self.current_action = action

        # Bewegung der Servos
        if action == 0:  # Links
            self.servo_position[0] = max(0, self.servo_position[0] - self.step_size)
        elif action == 1:  # Rechts
            self.servo_position[0] = min(180, self.servo_position[0] + self.step_size)
        elif action == 2:  # Hoch
            self.servo_position[1] = max(0, self.servo_position[1] - self.step_size)
        elif action == 3:  # Runter
            self.servo_position[1] = min(180, self.servo_position[1] + self.step_size)

        # Berechne LDR-Werte
        ldr_values = self._calculate_ldr_values()

        # Neue Belohnungsberechnung
        #reward, ldr_reward, balance_penalty, position_penalty = calculate_reward(ldr_values, self.servo_position, self.light_source)
        reward, intensity_bonus, spread_penalty = calculate_reward(ldr_values)
        reward = max(reward, 0)  # Reward bleibt mindestens bei 0

        # Speichere die aktuelle Aktion als vorherige Aktion
        self.previous_action = action

        done = reward > 3.5 or self._is_out_of_bounds()

        self._move_light_source()

        return ldr_values, reward, done, False, {}
    
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

        # Berechne LDR-Werte
        ldr_values = self._calculate_ldr_values()

        # Neue Belohnungsberechnung
        reward, ldr_reward, balance_penalty, position_penalty = calculate_reward(ldr_values, self.servo_position, self.light_source)
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
        distances = np.linalg.norm(self.light_source - self.ldr_positions, axis=1)  # Vektorisierte Berechnung
        intensities = np.clip(1 / (distances**2 + 1), 0, 1)
        return intensities / intensities.max() if len(intensities) > 1 and intensities.max() > 0 else intensities

    def _move_light_source(self, move_range=0.01, stationary=True):
        if not stationary:
            self.light_source += np.random.uniform(-move_range, move_range, size=2)
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