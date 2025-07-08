import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import cv2
import numpy as np
import time


"""
Reinforcement Learning Projekt: PPO-Agent für CarRacing-v3 (Box2D)

Dieses Notebook zeigt, wie ein RL-Agent mit PPO (Proximal Policy Optimization) trainiert wird,
um die komplexe Aufgabe des autonomen Fahrens in der Box2D CarRacing-v3 Umgebung zu lösen.

Das Notebook umfasst:
1. Umgebungsvorbereitung
2. PPO-Agenten-Konfiguration
3. Training des Agenten
4. Evaluation & Visualisierung
5. Automatische Endlos-Ausführung zur Live-Demonstration

Die Umgebung liefert pixelbasierte Beobachtungen (RGB-Bilder), welche mit einer CNN-Policy
verarbeitet werden. Ziel des Agenten ist es, die Rennstrecke effizient abzufahren.

Durch die Nutzung von Deep Reinforcement Learning mit Proximal Policy Optimization kann der Agent
lernen, komplexe Navigationsstrategien zu entwickeln, um Hindernissen auszuweichen und Kurven
optimal zu durchfahren.

Die CarRacing-Umgebung ist dabei ideal für kontinuierliche Steuerungsaufgaben geeignet und bietet
sowohl eine hohe Komplexität als auch eine gute Visualisierbarkeit des Lernprozesses.
"""

# Schritt 1: Umgebung vorbereiten
# Wir verwenden die neueste CarRacing-v3 Umgebung von OpenAI Gym.
# Diese Umgebung nutzt Box2D-Physik und ist eine Standard-Benchmark für RL in kontinuierlichen Steuerungsaufgaben.
# Beobachtungen bestehen aus 96x96 RGB-Bildern, die den aktuellen Zustand der Strecke darstellen.
# Der Aktionsraum ist kontinuierlich und umfasst Lenken, Gas geben und Bremsen.

env = make_vec_env("CarRacing-v3", n_envs=1)

# Schritt 2: PPO-Agent mit CNN-Policy definieren
# Der PPO-Agent nutzt eine Convolutional Neural Network Policy (CnnPolicy), um direkt aus Bilddaten zu lernen.
# Dies ermöglicht es dem Agenten, visuelle Informationen effektiv zu verarbeiten und darauf basierend Handlungen abzuleiten.
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_car_racing_tensorboard/"
)

# Schritt 3: Training
# Wir trainieren den Agenten für 500.000 Zeitschritte.
# Hinweis: Das Training kann je nach Hardware einige Zeit in Anspruch nehmen.
# In dieser Phase lernt der Agent durch Trial-and-Error, die Kontrolle über das Fahrzeug zu verbessern.
# Rewards werden vergeben, wenn der Agent auf der Strecke bleibt und möglichst schnell fährt.

print("\n=== Starte Training des PPO-Agenten für CarRacing-v3 ===")
model.learn(total_timesteps=500_000)
print("=== Training abgeschlossen ===\n")

# Modell speichern
model.save("ppo_car_racing")
print("Modell wurde gespeichert unter 'ppo_car_racing'\n")

# Schritt 4: Modell laden und Evaluation durchführen
# Hier zeigen wir, wie der Agent die Strecke befährt.
# Das Rendering erfolgt live im OpenCV-Fenster, damit das Verhalten sichtbar wird.

eval_env = gym.make("CarRacing-v3", render_mode="rgb_array")
model = PPO.load("ppo_car_racing")

cv2.namedWindow("Car Racing - Live Demo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Car Racing - Live Demo", 1024, 1024)

print("\n=== Starte Endlos-Demo des trainierten Agenten ===")

# Automatische Endlos-Ausführung zur Live-Demonstration
# Der Agent fährt kontinuierlich, Episoden starten nach jedem Absturz neu.
# Ziel: Live-Visualisierung des aktuellen Agentenverhaltens über mehrere Episoden hinweg.

while True:
    obs = eval_env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        # Aktion vom Agenten vorhersagen
        action, _ = model.predict(obs)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward
        steps += 1

        # Bild für OpenCV-Rendering vorbereiten
        frame = eval_env.render()

        # Zusätzliche HUD-Informationen ins Bild zeichnen
        hud = np.zeros((40, frame.shape[1], 3), dtype=np.uint8)
        text = f"Steps: {steps} | Reward: {total_reward:.2f}"
        cv2.putText(hud, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        combined = np.vstack([frame, hud])

        # Bild anzeigen
        cv2.imshow("Car Racing - Live Demo", combined)


        # Falls 'q' gedrückt wird, Programm beenden
        if cv2.waitKey(20) & 0xFF == ord("q"):
            eval_env.close()
            cv2.destroyAllWindows()
            print("\nDemo manuell beendet.\n")
            exit()

    # Nach jeder Episode Zusammenfassung ausgeben
    print(f"Episode abgeschlossen. Gesamt-Reward: {total_reward:.2f}, Schritte: {steps}")
    print("Starte neue Episode...\n")
    
