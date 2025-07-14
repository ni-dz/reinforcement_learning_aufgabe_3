import gymnasium as gym
from gymnasium.wrappers import (
    ResizeObservation,
    FrameStackObservation,
    GrayscaleObservation,
)
import torch
import torch.nn as nn
import numpy as np
import imageio


# Diskrete zu kontinuierliche Aktion (wie im Notebook)
def discrete_to_continuous_action(action):
    if action == 0:
        return np.array([0.0, 0.0, 0.0])  # Nichts tun
    elif action == 1:
        return np.array([1.0, 0.0, 0.0])  # Rechts
    elif action == 2:
        return np.array([-1.0, 0.0, 0.0])  # Links
    elif action == 3:
        return np.array([0.0, 1.0, 0.0])  # Gas
    elif action == 4:
        return np.array([0.0, 0.0, 0.8])  # Bremsen
    return np.array([0.0, 0.0, 0.0])


# SkipFrame Wrapper um mehrere Schritte zu überspringen
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


# DQN-Netzwerkarchitektur
class DQN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# Agent für Inferenz
class InferenceAgent:
    def __init__(self, model_path, action_dim=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DQN(action_dim).to(self.device)
        self.q_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.q_net.eval()

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()


# Test Funktion für den Agenten
def test_agent(
    model_path="trained_agent.pth", gif_path="agent_run.gif", max_steps=1000
):
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=4)
    env = SkipFrame(env, skip=4)

    agent = InferenceAgent(model_path)
    state, _ = env.reset()
    frames = []
    total_reward = 0
    step = 0

    while True:
        frame = env.render()
        frames.append(frame)

        action = agent.act(np.array(state))
        cont_action = discrete_to_continuous_action(action)
        state, reward, terminated, truncated, _ = env.step(cont_action)

        total_reward += reward
        step += 1
        if terminated or truncated or step >= max_steps:
            break

    env.close()
    imageio.mimsave(gif_path, frames, fps=30, loop=0)
    print(f"\nTest abgeschlossen — {step} Schritte")
    print(f"Gesamtreward: {total_reward:.2f}")
    print(f"GIF gespeichert: {gif_path}")


if __name__ == "__main__":
    test_agent()
