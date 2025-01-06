import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Funktion zur Auswahl und Anzeige der Heatmap
def display_heatmap(option, servo_positions, rewards, ldr_values, light_source_positions=None):
    if option == "1":
        # Heatmap der Servo-Positionen
        heatmap_data = pd.DataFrame({
            "X_Position": servo_positions[:, 0],
            "Y_Position": servo_positions[:, 1]
        })
        plt.figure(figsize=(10, 8))
        sns.histplot(data=heatmap_data, x="X_Position", y="Y_Position", bins=50, pthresh=0.01, cmap="viridis")
        plt.title("Heatmap der Servo-Positionen")
        plt.xlabel("Servo X-Position")
        plt.ylabel("Servo Y-Position")
        plt.colorbar(label="Häufigkeit")
        plt.show()

    elif option == "2":
        # Heatmap der Rewards
        reward_data = pd.DataFrame({
            "X_Position": servo_positions[:, 0],
            "Y_Position": servo_positions[:, 1],
            "Reward": rewards
        })
        pivot_table = reward_data.pivot_table(values="Reward", index="Y_Position", columns="X_Position", aggfunc="mean")
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, cmap="viridis", cbar_kws={"label": "Belohnung"})
        plt.title("Heatmap der Rewards")
        plt.xlabel("Servo X-Position")
        plt.ylabel("Servo Y-Position")
        plt.show()

    elif option == "3" and light_source_positions is not None:
        # Heatmap der Distanzen zur Lichtquelle
        distances = np.linalg.norm(servo_positions - light_source_positions * 180, axis=1)
        distance_data = pd.DataFrame({
            "X_Position": servo_positions[:, 0],
            "Y_Position": servo_positions[:, 1],
            "Distance": distances
        })
        pivot_table = distance_data.pivot_table(values="Distance", index="Y_Position", columns="X_Position", aggfunc="mean")
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, cmap="viridis", cbar_kws={"label": "Distanz zur Lichtquelle"})
        plt.title("Heatmap der Distanz zur Lichtquelle")
        plt.xlabel("Servo X-Position")
        plt.ylabel("Servo Y-Position")
        plt.show()

    elif option == "4":
        # Heatmap der LDR-Werte
        ldr_data = pd.DataFrame({
            "X_Position": servo_positions[:, 0],
            "Y_Position": servo_positions[:, 1],
            "LDR_Value": np.mean(ldr_values, axis=1)
        })
        pivot_table = ldr_data.pivot_table(values="LDR_Value", index="Y_Position", columns="X_Position", aggfunc="mean")
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, cmap="viridis", cbar_kws={"label": "LDR-Wert"})
        plt.title("Heatmap der LDR-Werte")
        plt.xlabel("Servo X-Position")
        plt.ylabel("Servo Y-Position")
        plt.show()

    else:
        print("Ungültige Auswahl. Bitte starte das Programm erneut und wähle eine gültige Option.")

# Hauptprogramm
if __name__ == "__main__":
    # Lade die gespeicherten Logs
    data = np.load("logs_512_256_128.npz")
    servo_positions = data["servo_positions"]
    rewards = data["rewards"]
    ldr_values = data["ldr_values"]
    light_source_positions = data.get("light_source_positions")  # Optional, falls nicht vorhanden

    # Menü anzeigen
    print("Wähle die gewünschte Heatmap:")
    print("1: Heatmap der Servo-Positionen")
    print("2: Heatmap der Rewards")
    print("3: Heatmap der Distanz zur Lichtquelle")
    print("4: Heatmap der LDR-Werte")
    option = input("Eingabe (1, 2, 3 oder 4): ").strip()

    # Heatmap anzeigen
    display_heatmap(option, servo_positions, rewards, ldr_values, light_source_positions)
