# main.py
import gymnasium as gym
import numpy as np
import cv2
from dqn_agent import DQNAgent
from utils import setup_live_plot, update_plot, moving_average
import matplotlib.pyplot as plt
import torch

env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
agent = DQNAgent(action_dim=5)
num_episodes = 500
# frame_skip = 4  # Originalwert
frame_skip = 8
# step_limit = 1000
step_limit = 500  # Reduzierte Schrittzahl pro Episode für schnellere Ausführung

fig, ax, reward_line, loss_line, epsilon_line, rewards_list, losses_list, epsilon_list = setup_live_plot()

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    step_count = 0
    frame_count = 0

    while not done and step_count < step_limit:
        img = env.render()
        cv2.imshow("CarRacing", img)
        if cv2.waitKey(1) == ord('q'):
            done = True
            break

        state_gray = cv2.resize(state, (96, 96))
        state_gray = np.transpose(state_gray, (2, 0, 1))

        if frame_count % frame_skip == 0:
            action = agent.select_action(state_gray)
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Hier zusätzliche Zeitstrafe einfügen:
        reward -= 0.5  # oder -1 für stärkere Bestrafung

        done = terminated or truncated


        # Reset, wenn Agent gegen die Wand fährt (Reward < -20 als Beispielschwelle)
        if reward < -20:
            done = True
            reward -= 100  # Extra-Penalty zur Bestrafung

        next_state_gray = cv2.resize(next_state, (96, 96))
        next_state_gray = np.transpose(next_state_gray, (2, 0, 1))

        agent.replay_buffer.push(state_gray, action, reward, next_state_gray, done)
        state = next_state

        loss = agent.train_step()
        episode_reward += reward

        frame_count += 1
        step_count += 1

    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

    rewards_list.append(episode_reward)
    losses_list.append(loss)
    epsilon_list.append(agent.epsilon)

    update_plot(rewards_list, losses_list, epsilon_list, reward_line, loss_line, epsilon_line, ax)

    ma_reward = moving_average(rewards_list, window_size=10)
    print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, MA10: {ma_reward[-1]:.2f}, "
          f"Epsilon: {agent.epsilon:.3f}, Loss: {loss:.4f}")

env.close()
cv2.destroyAllWindows()

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

# Agent speichern
torch.save(agent.q_net.state_dict(), "trained_agent.pth")
print("Agent saved as 'trained_agent.pth'")
