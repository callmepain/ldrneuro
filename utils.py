import numpy as np
import pandas as pd
import os
import torch

def calculate_reward(
    ldr_values, servo_position, light_source, previous_reward=0, previous_action=None, current_action=None
):
    # Schwellenwert für "starke" LDR-Werte
    strong_ldr_threshold = 0.8  # Werte über 0.8 gelten als "stark"
    
    # Anzahl der "starken" LDRs
    num_strong_ldrs = sum(1 for value in ldr_values if value > strong_ldr_threshold)
    
    # Bonus für mehrere LDRs im Licht
    multi_ldr_bonus = 0
    if num_strong_ldrs == 2:
        multi_ldr_bonus = 1.0  # Bonus für 2 starke LDRs
    elif num_strong_ldrs == 3:
        multi_ldr_bonus = 2.0  # Bonus für 3 starke LDRs
    elif num_strong_ldrs == 4:
        multi_ldr_bonus = 4.0  # Bonus für alle 4 starke LDRs
    
    # Belohnung basierend auf den LDR-Werten
    spread_penalty = 1.0 * np.std(ldr_values)  # Bestrafung für ungleichmäßige Werte
    intensity_bonus = (np.mean(ldr_values) ** 4) if np.mean(ldr_values) > 0 else 0
    ldr_reward = sum([2.0 * v for v in ldr_values]) + intensity_bonus - spread_penalty + multi_ldr_bonus

    # Belohnung auf 0 setzen, wenn kein Licht erkannt wird
    if np.mean(ldr_values) <= 0.4:  # Erhöhe den Schwellenwert
        ldr_reward = 0

    # Positionsstrafe
    relative_distance = np.linalg.norm(np.array(servo_position) - light_source * 180) / 180

    position_penalty = 0.5 * (relative_distance ** 2) * (1 + relative_distance)
    # Härtere Bestrafung für größere Distanzen
    if relative_distance > 0.6:
        position_penalty = 1.5 * (relative_distance ** 4) * (1 + relative_distance)
    elif relative_distance > 0.3:
        position_penalty = 1.0 * (relative_distance ** 3) * (1 + relative_distance)
    else:
        position_penalty = 0.7 * (relative_distance ** 2) * (1 + relative_distance)

    # Zusätzliche Strafe bei schwachem Licht
    if np.mean(ldr_values) < 0.4:  # Schwellenwert für schwaches Licht
        position_penalty *= 5  # Drastischere Strafe

    # LDR-Reward abhängig von Position
    ldr_reward *= max(0, 1 - relative_distance / 1.3)  # Strengere Skalierung
    
    # Balancing-Strafe
    balance_penalty = min(1.0, 1.0 * np.std(ldr_values))
    if np.mean(ldr_values) < 0.4:  # Schwellenwert für schwaches Licht
        balance_penalty += 1.5
    balance_penalty *= (1 + relative_distance)  # Abhängig von der Distanz

    # Gesamtreward
    reward = ldr_reward - position_penalty - balance_penalty

    raw_reward = ldr_reward - position_penalty - balance_penalty
    # Glätten und Begrenzen des Rewards
    reward = 0.6 * previous_reward + 0.4 * reward
    if np.mean(ldr_values) > 0.95 and relative_distance < 0.1:
        reward = 4.0
    reward = np.clip(reward, -5, 5)

    return reward, ldr_reward, balance_penalty, position_penalty, raw_reward


def calculate_reward_alt_und_gut(
    ldr_values, servo_position, light_source, previous_reward=0, previous_action=None, current_action=None
):
    # LDR-Belohnung
    spread_penalty = 1.0 * np.std(ldr_values)
    intensity_bonus = (np.mean(ldr_values) ** 5) if np.mean(ldr_values) > 0 else 0
    ldr_reward = sum([3.0 * v for v in ldr_values]) + intensity_bonus - spread_penalty

    # Belohnung auf 0 setzen, wenn kein Licht erkannt wird
    if np.mean(ldr_values) <= 0.4:  # Erhöhe den Schwellenwert
        ldr_reward = 0

    # Positionsstrafe
    relative_distance = np.linalg.norm(np.array(servo_position) - light_source * 180) / 180

    position_penalty = 0.5 * (relative_distance ** 2) * (1 + relative_distance)
    # Härtere Bestrafung für größere Distanzen
    if relative_distance > 0.6:
        position_penalty = 1.5 * (relative_distance ** 4) * (1 + relative_distance)
    elif relative_distance > 0.3:
        position_penalty = 1.0 * (relative_distance ** 3) * (1 + relative_distance)
    else:
        position_penalty = 0.7 * (relative_distance ** 2) * (1 + relative_distance)

    # Zusätzliche Strafe bei schwachem Licht
    if np.mean(ldr_values) < 0.4:  # Schwellenwert für schwaches Licht
        position_penalty *= 5  # Drastischere Strafe

    # LDR-Reward abhängig von Position
    ldr_reward *= max(0, 1 - relative_distance / 1.3)  # Strengere Skalierung
    
    # Balancing-Strafe
    balance_penalty = min(0.8, 0.8 * np.std(ldr_values))
    if np.mean(ldr_values) < 0.4:  # Schwellenwert für schwaches Licht
        balance_penalty += 1.5
    balance_penalty *= (1 + relative_distance)  # Abhängig von der Distanz

    # Gesamtreward
    reward = ldr_reward - position_penalty - balance_penalty

    raw_reward = ldr_reward - position_penalty - balance_penalty
    # Glätten und Begrenzen des Rewards
    reward = 0.6 * previous_reward + 0.4 * reward
    if np.mean(ldr_values) > 0.95 and relative_distance < 0.1:
        reward = 4.0
    reward = np.clip(reward, -5, 5)

    return reward, ldr_reward, balance_penalty, position_penalty, raw_reward


def calculate_reward_alt(
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

    raw_reward = ldr_reward - position_penalty - balance_penalty

    # Glätten und Begrenzen des Rewards
    reward = 0.6 * previous_reward + 0.4 * reward
    reward = np.clip(reward, -5, 5)

    return reward, ldr_reward, balance_penalty, position_penalty, raw_reward