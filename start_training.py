import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
import torch
import os
import time
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import calculate_reward
from environment import LightTrackingEnv
from callbacks import DebugCallback
from stable_baselines3.common.utils import set_random_seed

print(torch.cuda.is_available())  # Sollte True sein

def make_env(env_id, rank, seed=0):
    """
    Fabrikfunktion, die eine neue Instanz der Umgebung erzeugt.
    :param env_id: ID der Umgebung (kann ignoriert werden).
    :param rank: Der Rang der Umgebung.
    :param seed: Der globale Seed.
    """
    def _init():
        env = LightTrackingEnv()
        env.seed(seed + rank)  # Seed für jede Instanz
        return env
    return _init

# Updated train_model function to ensure unique logging for each retraining session
def train_model(seed=42):
    set_random_seed(seed)
    print("Möchtest du ein neues Modell erstellen oder ein vorhandenes laden?")
    print("1: Neues Modell erstellen")
    print("2: Vorhandenes Modell laden")

    choice = input("Eingabe (1 oder 2): ").strip()

    if choice == "1":
        # Wähle die Netzwerkarchitektur
        print("Wähle die Netzwerkarchitektur:")
        print("1: Klein (256, 128)")
        print("2: Mittel (512, 256, 128)")
        print("3: Groß (1024, 512, 256)")
        print("4: Sehr groß (2048, 1024, 512, 256)")
        print("5: Sehr klein ESP32 (64, 32)")

        arch_choice = input("Eingabe (1, 2, 3 oder 4): ").strip()

        # Architektur basierend auf Eingabe
        if arch_choice == "1":
            policy_kwargs = dict(net_arch=[256, 128])
            model_suffix = "256_128"
        elif arch_choice == "2":
            policy_kwargs = dict(net_arch=[512, 256, 128])
            model_suffix = "512_256_128"
        elif arch_choice == "3":
            policy_kwargs = dict(net_arch=[1024, 512, 256])
            model_suffix = "1024_512_256"
        elif arch_choice == "4":
            policy_kwargs = dict(net_arch=[2048, 1024, 512, 256])
            model_suffix = "2048_1024_512_256"
        elif arch_choice == "5":
            policy_kwargs = dict(net_arch=[64, 32])
            model_suffix = "64_32"
        else:
            print("Ungültige Auswahl! Standardarchitektur wird verwendet.")
            policy_kwargs = dict(net_arch=[2048, 1024, 512])
            model_suffix = "2048_1024_512"

        # Modellpfad und Logs
        model_path = f"light_tracking_ppo_{model_suffix}"
        log_dir = f"./light_tracking_tensorboard/{model_suffix}/session_{int(time.time())}/"
        os.makedirs(log_dir, exist_ok=True)

        print(f"Erstelle neues Modell: {model_path}...")
        env = DummyVecEnv([lambda: LightTrackingEnv() for _ in range(16)])  # 24 parallele Threads
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=0.0001,
            n_steps=8192,
            batch_size=2048,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.9,
            clip_range=0.1,
            verbose=2,
            device="cuda" if torch.cuda.is_available() else "cpu",  # GPU bevorzugen
            tensorboard_log=log_dir
        )

    elif choice == "2":
        # Wähle die Architektur des Modells, das geladen werden soll
        print("Wähle die Netzwerkarchitektur des Modells:")
        print("1: Klein (256, 128)")
        print("2: Mittel (512, 256, 128)")
        print("3: Groß (1024, 512, 256)")
        print("4: Sehr groß (2048, 1024, 512, 256)")
        print("5: Sehr klein ESP32 (64, 32)")

        arch_choice = input("Eingabe (1, 2, 3 oder 4): ").strip()

        if arch_choice == "1":
            model_suffix = "256_128"
        elif arch_choice == "2":
            model_suffix = "512_256_128"
        elif arch_choice == "3":
            model_suffix = "1024_512_256"
        elif arch_choice == "4":
            model_suffix = "2048_1024_512_256"
        elif arch_choice == "5":
            model_suffix = "64_32"
        else:
            print("Ungültige Auswahl! Standardmodell wird verwendet.")
            model_suffix = "2048_1024_512"

        # Modellpfad
        model_path = f"light_tracking_ppo_{model_suffix}"
        log_dir = f"./light_tracking_tensorboard/{model_suffix}/session_{int(time.time())}/"
        os.makedirs(log_dir, exist_ok=True)

        if not os.path.exists(model_path + ".zip"):
            print(f"Kein gespeichertes Modell gefunden: {model_path}.")
            return

        print(f"Bestehendes Modell gefunden: {model_path}. Lade Modell...")
        model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")  # GPU bevorzugen
        env = DummyVecEnv([lambda: LightTrackingEnv() for _ in range(16)])  # 64 Threads
        model.set_env(env)

        # Abfrage, ob Batchgröße geändert werden soll
        change_batch_size = input("Möchtest du die Batchgröße ändern? (ja/nein): ").strip().lower()
        if change_batch_size == "ja":
            try:
                # Eingabe der neuen Batchgröße mit Validierung
                new_batch_size = int(input("Neue Batchgröße eingeben (z. B. 64, 128, 256): ").strip())
                if new_batch_size <= 0:
                    raise ValueError("Die Batchgröße muss eine positive Zahl sein.")
                print(f"Neue Batchgröße wird auf {new_batch_size} gesetzt.")

                # Neues Modell mit geänderter Batchgröße erstellen
                new_model = PPO(
                    "MlpPolicy",
                    env,
                    policy_kwargs=model.policy_kwargs,
                    learning_rate=model.learning_rate,
                    n_steps=model.n_steps,
                    batch_size=new_batch_size,
                    gamma=model.gamma,
                    gae_lambda=model.gae_lambda,
                    clip_range=model.clip_range,
                    verbose=model.verbose,
                    device="cuda" if torch.cuda.is_available() else "cpu",  # GPU bevorzugen
                    tensorboard_log=model.tensorboard_log
                )

                # Alte Policy übertragen
                new_model.policy.load_state_dict(model.policy.state_dict())
                model = new_model
                print("Batchgröße erfolgreich geändert und neues Modell erstellt.")
            except ValueError as e:
                print(f"Fehler bei der Eingabe der Batchgröße: {e}")

    else:
        print("Ungültige Auswahl!")
        return

    # Logger konfigurieren
    logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)

    # Training starten
    debug_callback = DebugCallback()
    start_time = time.time()
    model.learn(total_timesteps=1_000_000, tb_log_name=f"training_{model_suffix}_{int(time.time())}", callback=debug_callback)
    model.save(model_path)
    print(f"Modell gespeichert als '{model_path}'.")
    print(f"TensorBoard Logs wurden in '{log_dir}' gespeichert.")
     # Zugriff auf die ursprüngliche Umgebung und Logs speichern
    env_instance = env.envs[0]  # Greife auf die erste Umgebung im DummyVecEnv zu
    env_instance.save_logs(f"logs_{model_suffix}.npz")  # Logs speichern
    end_time = time.time()
    print(f"Trainingszeit für {model_suffix}: {end_time - start_time} Sekunden")

def test_model():
    print("Wähle die Netzwerkarchitektur des Modells:")
    print("1: Klein (256, 128)")
    print("2: Mittel (512, 256, 128)")
    print("3: Groß (1024, 512, 256)")
    print("4: Sehr groß (2048, 1024, 512, 256)")
    print("5: Sehr klein ESP32 (64, 32)")

    arch_choice = input("Eingabe (1, 2, 3, 4 oder 5): ").strip()

    if arch_choice == "1":
        model_suffix = "256_128"
    elif arch_choice == "2":
        model_suffix = "512_256_128"
    elif arch_choice == "3":
        model_suffix = "1024_512_256"
    elif arch_choice == "4":
        model_suffix = "2048_1024_512_256"
    elif arch_choice == "5":
        model_suffix = "64_32"
    else:
        print("Ungültige Auswahl! Standardmodell wird verwendet.")
        model_suffix = "2048_1024_512"

    model_path = f"light_tracking_ppo_{model_suffix}"

    if not os.path.exists(model_path + ".zip"):
        print(f"Kein gespeichertes Modell gefunden: {model_path}.")
        return

    print(f"Lade Modell: {model_path}...")
    model = PPO.load(model_path)
    env = LightTrackingEnv()
    obs, _ = env.reset()

    for step in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        print(f"Step: {step}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Servo Position: {env.servo_position}")
        print(f"LDR Values: {obs}")
        print("-" * 40)

        if done:
            print("Optimaler Zustand erreicht!")
            break

if __name__ == "__main__":
    print("Wähle einen Modus:")
    print("1: Trainieren")
    print("2: Testen")

    choice = input("Eingabe (1 oder 2): ").strip()
    if choice == "1":
        train_model()
    elif choice == "2":
        test_model()
    else:
        print("Ungültige Auswahl!")
