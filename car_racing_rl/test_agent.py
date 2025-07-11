# test_agent.py
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation
from gymnasium.wrappers.stateful_observation import FrameStackObservation
from gymnasium.wrappers.transform_observation import GrayscaleObservation
import numpy as np
import torch
import time
from dqn_agent import DQNAgent

def discrete_to_continuous_action(action):
    if action == 0:
        return np.array([0.0, 0.0, 0.0])
    elif action == 1:
        return np.array([-1.0, 0.0, 0.0])
    elif action == 2:
        return np.array([1.0, 0.0, 0.0])
    elif action == 3:
        return np.array([0.0, 1.0, 0.0])
    elif action == 4:
        return np.array([0.0, 0.0, 0.8])
    else:
        return np.array([0.0, 0.0, 0.0])

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
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

# Environment (gleich wie im Training!)
env = gym.make("CarRacing-v3", render_mode="human")
env = GrayscaleObservation(env, keep_dim=False)
env = ResizeObservation(env, (84, 84))
env = FrameStackObservation(env, stack_size=4)
env = SkipFrame(env, skip=4)

# Agent laden
agent = DQNAgent(action_dim=5)
agent.q_net.load_state_dict(torch.load("trained_agent.pth"))
agent.q_net.eval()

print("🚗 Agent Test gestartet...")

num_test_episodes = 3  # Teste 3 Fahrten

for episode in range(num_test_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        time.sleep(0.01)  # Optional, um die Fahrt lesbar zu machen

        # WICHTIG: Im Test keine Exploration!
        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.q_net(state_tensor)
        action = q_values.argmax().item()
        continuous_action = discrete_to_continuous_action(action)

        next_state, reward, terminated, truncated, _ = env.step(continuous_action)
        total_reward += reward
        state = next_state
        done = terminated or truncated

    print(f"Test-Episode {episode + 1}: Reward = {total_reward:.2f}")

env.close()
print("✅ Test abgeschlossen.")
