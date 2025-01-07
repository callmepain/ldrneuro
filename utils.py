import numpy as np
import pandas as pd
import os

def calculate_reward(
    ldr_values, servo_position, light_source, previous_reward=0, previous_action=None, current_action=None
):
    # LDR-Belohnung
    spread_penalty = 1.5 * np.std(ldr_values)
    intensity_bonus = (np.mean(ldr_values) ** 2) if np.mean(ldr_values) > 0 else 0  # Bonus nur bei erfasstem Licht
    ldr_reward = sum([1.0 * v for v in ldr_values]) + intensity_bonus - spread_penalty

    # Belohnung auf 0 setzen, wenn kein Licht erkannt wird
    if np.mean(ldr_values) <= 0.05:  # Schwellenwert für "kein Licht"
        ldr_reward = 0

    # Positionsstrafe
    relative_distance = np.linalg.norm(np.array(servo_position) - light_source * 180) / 180

    # Skalierung des LDR-Rewards basierend auf der Distanz zur Lichtquelle
    ldr_reward *= max(0, 1 - relative_distance / 2)  # Weniger drastisch

    position_penalty = 0.6 * (relative_distance ** 2) * (1 + relative_distance)
    if relative_distance > 0.6:
        position_penalty *= 2
    elif relative_distance > 0.3:
        position_penalty *= 1.5
    if np.mean(ldr_values) < 0.4:  # Strafe bei schwachem Licht
        position_penalty *= 3

    # LDR-Reward abhängig von Position
    ldr_reward *= max(0, 1 - relative_distance)
    
    # Balance-Strafe
    balance_penalty = 1.2 * np.std(ldr_values)
    if previous_action != current_action:
        balance_penalty += 0.1 * abs(previous_action - current_action)

    # Gesamtreward
    reward = ldr_reward - position_penalty - balance_penalty

    # Glätten und Begrenzen des Rewards
    reward = 0.6 * previous_reward + 0.4 * reward
    reward = np.clip(reward, -5, 5)

    return reward, ldr_reward, balance_penalty, position_penalty
