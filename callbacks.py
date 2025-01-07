from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from utils import calculate_reward
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import threading

class DebugCallback(BaseCallback):
    def __init__(self, update_interval=10, enable_visualization=False):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

        self.update_interval = update_interval
        self.enable_visualization = enable_visualization  # Visualisierung aktivieren/deaktivieren

        if self.enable_visualization:
            self.fig, self.ax = plt.subplots()  # Initialisiere Matplotlib-Figur und Achsen
            self.light_source_point, = self.ax.plot([], [], 'yo', label='Light Source')
            self.servo_point, = self.ax.plot([], [], 'ro', label='Servo Position')

    def _on_step(self) -> bool:
        # Zugriff auf die ursprüngliche Umgebung
        env = self.training_env.envs[0].unwrapped
        # Radien aus der Umgebung verwenden
        light_radius = env.light_radius
        ldr_radius = env.sensor_radius

        # Aktualisiere die Positionen dynamisch aus der Umgebung
        self.light_source_position = [env.light_source[0] * 180, env.light_source[1] * 180]
        self.servo_position = [env.servo_position[0], env.servo_position[1]]

        # Aktualisiere die Visualisierung nur alle N Schritte
        if self.enable_visualization and self.n_calls % self.update_interval == 0:
            self.ax.clear()

            # Lichtquelle und Radius visualisieren
            self.ax.add_patch(
                plt.Circle((self.light_source_position[0], self.light_source_position[1]),
                        light_radius, color='yellow', alpha=0.5, label='Light Radius')
            )
            self.ax.plot(self.light_source_position[0], self.light_source_position[1], 'yo', label='Light Source')
            
            # Servo-Position und LDR-Aufnahme-Radius visualisieren
            self.ax.add_patch(
                plt.Circle((self.servo_position[0], self.servo_position[1]),
                        ldr_radius, color='red', alpha=0.5, label='LDR Radius')
            )
            self.ax.plot(self.servo_position[0], self.servo_position[1], 'ro', label='Servo Position')

            # Achsen und Legende konfigurieren
            self.ax.set_xlim(0, max(180, light_radius + 20))
            self.ax.set_ylim(0, max(180, light_radius + 20))
            self.ax.set_title('Training Visualization')
            self.ax.set_xlabel('X Position')
            self.ax.set_ylabel('Y Position')
            self.ax.legend()

            # Zeichne die Aktualisierungen
            plt.pause(0.01)

        # LDR-Werte berechnen
        ldr_values = env._calculate_ldr_values()

        # Belohnungslogik
        reward, ldr_reward, balance_penalty, position_penalty = calculate_reward(
            ldr_values, env.servo_position, env.light_source
        )
        #reward, intensity_bonus, spread_penalty = calculate_reward(ldr_values)

        # Logge zusätzliche Metriken in TensorBoard
        self.logger.record("custom/servo_position_x", env.servo_position[0])
        self.logger.record("custom/servo_position_y", env.servo_position[1])
        self.logger.record("custom/reward", reward)
        self.logger.record("custom/ldr_reward", ldr_reward)
        self.logger.record("custom/balance_penalty", balance_penalty)
        self.logger.record("custom/position_penalty", position_penalty)
        self.logger.record("custom/light_source_x", env.light_source[0])
        self.logger.record("custom/light_source_y", env.light_source[1])

        # Logge Modellgewichte (Mittelwert und maximale Änderung)
        if self.n_calls % 100 == 0:  # Alle 100 Schritte
            print(f"Step {self.n_calls}:")
            print(f"  Servo Position: {env.servo_position}")
            print(f"  Reward: {reward}")
            print(f"  Modellgewicht (erste Layer): {list(self.model.policy.parameters())[0].mean().item():.6f}")
            with torch.no_grad():
                for name, param in self.model.policy.named_parameters():
                    self.logger.record(f"weights/layer0_mean", param.mean().item())
                    self.logger.record(f"weights/layer0_max", param.max().item()) 
                    if param.grad is not None:
                        self.logger.record(f"weights/layer0_mean", param.mean().item())
                        self.logger.record(f"weights/layer0_max", param.max().item())   

        # Episode abgeschlossen?
        if self.locals["dones"][0]:
            self.episode_rewards.append(sum(self.locals["rewards"]))
            self.episode_lengths.append(len(self.locals["rewards"]))

            # Logge Episodenmetriken
            self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths))
            self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards))

        # Logs schreiben (zwingt TensorBoard zur Verarbeitung)
        if self.n_calls % 100 == 0:  # Alle 100 Schritte
            self.logger.dump(self.num_timesteps)

        return True  # Training fortsetzen
