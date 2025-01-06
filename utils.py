import numpy as np
import pandas as pd
import os

def calculate_reward_alt(ldr_values, servo_position, light_source, previous_reward=0):
    # LDR-Belohnung (leicht reduziert)
    #ldr_reward = sum([4 * v for v in ldr_values]) #vorher 5
    
    #ldr_reward = 4.0 * min(ldr_values) + 2.0 * np.mean(ldr_values)
    
    spread_penalty = 0.5 * np.std(ldr_values)
    ldr_reward = sum([4 * v for v in ldr_values]) - spread_penalty
    
    # Balance-Strafe (leicht reduziert)
    balance_penalty = 0.25 * np.std(ldr_values) #alt 0.15
    
    # Positionsstrafe, abhängig von der Distanz zur Lichtquelle
    relative_distance = np.linalg.norm(np.array(servo_position) - light_source * 180) / 180
    position_penalty = 0.2 * (relative_distance ** 2) #alt 0.1
    
    # Verstärkung der Positionsstrafe, wenn Lichtstärke zu niedrig
    if np.mean(ldr_values) < 0.3:  # Beispiel-Schwelle für schwaches Licht
        position_penalty *= 2  # Strafe verdoppeln
    
    # Gesamtreward ohne Glättung
    reward = ldr_reward - balance_penalty - position_penalty
    
    # Optional: leichtes Smoothing des Rewards
    reward = 0.7 * previous_reward + 0.3 * reward #alt 0.8 0.2
    
    # Belohnung begrenzen
    reward = np.clip(reward, -5, 5)
    
    return reward, ldr_reward, balance_penalty, position_penalty

def calculate_reward_nicht_so_doll(ldr_values, servo_position, light_source, previous_reward=0, previous_action=None, current_action=None):
    # LDR Reward
    spread_penalty = 0.5 * np.std(ldr_values)
    ldr_reward = sum([4 * v * (1.2 if v > 0.7 else 1.0) for v in ldr_values]) - spread_penalty

    # Position Penalty
    relative_distance = np.linalg.norm(np.array(servo_position) - light_source * 180) / 180
    position_penalty = 0.2 * (relative_distance ** 2)
    if np.mean(ldr_values) < 0.3:
        position_penalty *= 2

    # Balance Penalty
    balance_penalty = 0.25 * np.std(ldr_values)
    if previous_action != current_action:
        balance_penalty += 0.05

    # Total Reward
    reward = ldr_reward - balance_penalty - position_penalty
    reward = 0.7 * previous_reward + 0.3 * reward
    reward = np.clip(reward, -5, 5)

    return reward, ldr_reward, balance_penalty, position_penalty

def calculate_reward_suboptiomal(
    ldr_values, servo_position, light_source, previous_reward=0, previous_action=None, current_action=None
):
    # Hauptziel: LDR-Belohnung
    spread_penalty = 1.0 * np.std(ldr_values)  # Strafe für Ungleichmäßigkeit
    intensity_bonus = (np.mean(ldr_values) ** 2) - 0.5 * np.std(ldr_values)  # Kombinierte Belohnung
    ldr_reward = sum([2 * v for v in ldr_values]) + intensity_bonus - spread_penalty

    # Positionsstrafe
    relative_distance = np.linalg.norm(np.array(servo_position) - light_source * 180) / 180
    position_penalty = 0.4 * (relative_distance ** 2)
    if relative_distance > 0.6:
        position_penalty *= 2
    elif relative_distance > 0.3:
        position_penalty *= 1.5
    if np.mean(ldr_values) < 0.4:  # Strafe bei schwachem Licht
        position_penalty *= 3

    # Balance-Strafe
    balance_penalty = 1.5 * np.std(ldr_values)
    if previous_action != current_action:
        balance_penalty += 0.1 * abs(previous_action - current_action)

    # Gesamtreward
    reward = ldr_reward - position_penalty - balance_penalty

    # Glätten und Begrenzen des Rewards
    reward = 0.6 * previous_reward + 0.4 * reward
    reward = np.clip(reward, -5, 5)

    return reward, ldr_reward, balance_penalty, position_penalty

def calculate_reward_recht_ok(
    ldr_values, servo_position, light_source, previous_reward=0, previous_action=None, current_action=None
):
    # LDR-Belohnung
    spread_penalty = 2.0 * np.std(ldr_values)  # Strafe für Ungleichmäßigkeit
    intensity_bonus = (np.mean(ldr_values) ** 2) if np.mean(ldr_values) > 0 else 0  # Bonus nur bei erfasstem Licht
    ldr_reward = sum([1.0 * v for v in ldr_values]) + intensity_bonus - spread_penalty

    # Belohnung auf 0 setzen, wenn kein Licht erkannt wird
    if np.mean(ldr_values) <= 0.05:  # Schwellenwert für "kein Licht"
        ldr_reward = 0

    # Positionsstrafe
    relative_distance = np.linalg.norm(np.array(servo_position) - light_source * 180) / 180
    position_penalty = 0.6 * (relative_distance ** 2)
    if relative_distance > 0.6:
        position_penalty *= 2
    elif relative_distance > 0.3:
        position_penalty *= 1.5
    if np.mean(ldr_values) < 0.4:  # Strafe bei schwachem Licht
        position_penalty *= 3

    # LDR-Reward abhängig von Position
    ldr_reward *= max(0, 1 - relative_distance)
    
    # Balance-Strafe
    balance_penalty = 1.5 * np.std(ldr_values)
    if previous_action != current_action:
        balance_penalty += 0.1 * abs(previous_action - current_action)

    # Gesamtreward
    reward = ldr_reward - position_penalty - balance_penalty

    # Glätten und Begrenzen des Rewards
    reward = 0.6 * previous_reward + 0.4 * reward
    reward = np.clip(reward, -5, 5)

    return reward, ldr_reward, balance_penalty, position_penalty

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
