from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from utils import calculate_reward
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class DebugCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

        self.figure, self.ax = plt.subplots()
        self.light_source_point, = self.ax.plot([], [], 'yo', label='Light Source')
        self.servo_point, = self.ax.plot([], [], 'ro', label='Servo Position')

        # Setup plot
        self.ax.set_xlim(0, 180)
        self.ax.set_ylim(0, 180)
        self.ax.legend()
        self.ax.set_title("Training Visualization")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")

    def _on_step(self) -> bool:
        # Zugriff auf die ursprüngliche Umgebung
        env = self.training_env.envs[0].unwrapped

        # Update positions
        self.light_source_point.set_data([env.light_source[0] * 180], [env.light_source[1] * 180])
        self.servo_point.set_data([env.servo_position[0]], [env.servo_position[1]])
        plt.pause(0.01)  # Update the plot

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
