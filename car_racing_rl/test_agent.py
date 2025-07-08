import gymnasium as gym
import numpy as np
import cv2
import torch
from dqn_agent import DQN  # Wichtig: Nutze exakt dasselbe DQN aus deinem Agenten-Code

# Environment laden
env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")

# Modell initialisieren
model = DQN(action_dim=5)
model.load_state_dict(torch.load("trained_agent.pth"))  # Laden der Gewichte
model.eval()  # In Evaluierungsmodus schalten (kein Training)

# Test-Loop (ohne Lernen)
state, _ = env.reset()
done = False

while not done:
    img = env.render()
    cv2.imshow("CarRacing Test", img)
    if cv2.waitKey(1) == ord('q'):
        break

    state_gray = cv2.resize(state, (96, 96))
    state_gray = np.transpose(state_gray, (2, 0, 1))
    state_tensor = torch.FloatTensor(state_gray).unsqueeze(0)

    with torch.no_grad():
        action = model(state_tensor).argmax().item()

    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()
cv2.destroyAllWindows()
