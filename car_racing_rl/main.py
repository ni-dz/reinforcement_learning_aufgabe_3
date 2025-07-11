# main.py
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation
from gymnasium.wrappers.stateful_observation import FrameStackObservation
from gymnasium.wrappers.transform_observation import GrayscaleObservation

# from utils import GrayScaleObservation  # Wenn du den Wrapper in utils.py speicherst
import numpy as np
import cv2
from dqn_agent import DQNAgent
from utils import setup_live_plot, update_plot, moving_average
import matplotlib.pyplot as plt
import torch
import time

def discrete_to_continuous_action(action):
    """
    Convert discrete action to continuous action for CarRacing environment.
    Actions: [steering, gas, brake]
    """
    if action == 0:  # Do nothing
        return np.array([0.0, 0.0, 0.0])
    elif action == 1:  # Turn left
        return np.array([-1.0, 0.0, 0.0])
    elif action == 2:  # Turn right
        return np.array([1.0, 0.0, 0.0])
    elif action == 3:  # Gas
        return np.array([0.0, 1.0, 0.0])
    elif action == 4:  # Brake
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


env = gym.make("CarRacing-v3", render_mode="human")
env = GrayscaleObservation(env, keep_dim=False)
env = ResizeObservation(env, (84, 84))
env = FrameStackObservation(env, stack_size=4)
env = SkipFrame(env, skip=4)


agent = DQNAgent(action_dim=5)
num_episodes = 2000
episode_rewards = []
frame_skip = 4

fig, ax, reward_line, loss_line, epsilon_line, rewards_list, losses_list, epsilon_list = setup_live_plot()

print("Observation Space:", env.observation_space)

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    step_num = 0
    neg_count = 0

    start_time = time.time()
    show_render = (episode + 1) % 1 == 0  # Jede 2. Episode zeigen

    while not done:
        if show_render:
            env.render()  # Zeigt das Spiel-Fenster
            # Optional: Kleine Pause für bessere Sichtbarkeit
            time.sleep(0.01)

        action = agent.select_action(np.array(state))
        continuous_action = discrete_to_continuous_action(action)
        next_state, reward, terminated, truncated, info = env.step(continuous_action)

        if info.get("on_grass", False):
            reward -= 20  # Gras fahren = schlecht, aber nicht katastrophal

        if info.get("track_direction", 1) < 0:
            reward -= 50  # Falsche Richtung = absolut schlimm
            terminated = True


        episode_reward += reward
        step_num += 1

        # Abbruchbedingungen falls Agent nicht mehr vorwärts kommt
        if episode_reward < 0:
            break
        if step_num > 300:
            if reward < 0:
                neg_count += 1
            if neg_count >= 25:
                break

        agent.replay_buffer.push(np.array(state), action, reward, np.array(next_state), terminated)
        loss = agent.train_step()
        state = next_state
        done = terminated or truncated

    agent.decay_epsilon(episode, factor=0.7)

    end_time = time.time()
    episode_duration = end_time - start_time

    rewards_list.append(episode_reward)
    losses_list.append(loss)
    epsilon_list.append(agent.epsilon)

    update_plot(rewards_list, losses_list, epsilon_list, reward_line, loss_line, epsilon_line, ax)

    ma_reward = moving_average(rewards_list, window_size=10)
    print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | "
          f"MA10: {ma_reward[-1]:.2f} | Epsilon: {agent.epsilon:.3f}, Loss: {loss:.4f} | "
          f"Duration: {episode_duration:.2f} seconds")

    if (episode + 1) % 100 == 0:
        torch.save(agent.q_net.state_dict(), f"checkpoint_episode_{episode+1}.pth")

env.close()

plt.plot(rewards_list, label='Reward')
plt.plot(losses_list, label='Loss')
plt.plot(epsilon_list, label='Epsilon')
plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('Training Metrics')
plt.legend()
plt.savefig('training_metrics.png', dpi=300)
plt.close()
print("Plot saved as 'training_metrics.png'")

torch.save(agent.q_net.state_dict(), "trained_agent.pth")
print("Agent saved as 'trained_agent.pth'")
