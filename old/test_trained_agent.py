# test_trained_agent.py
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation
from gymnasium.wrappers.stateful_observation import FrameStackObservation
from gymnasium.wrappers.transform_observation import GrayscaleObservation
import numpy as np
import cv2
import torch
from dqn_agent import DQNAgent
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

def test_agent(model_path="trained_agent.pth", num_episodes=5):
    """
    Test the trained agent
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to test
    """
    # Create environment with same wrappers as training
    env = gym.make("CarRacing-v3", render_mode="human")  # "human" for visualization
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=4)
    env = SkipFrame(env, skip=4)
    
    # Create agent
    agent = DQNAgent(action_dim=5)
    
    # Load trained model
    try:
        agent.q_net.load_state_dict(torch.load(model_path, map_location=agent.device))
        agent.q_net.eval()  # Set to evaluation mode
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found!")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set epsilon to 0 for pure exploitation (no random actions)
    agent.epsilon = 0.0
    
    print(f"Testing agent for {num_episodes} episodes...")
    print("Press 'q' to quit early, 'ESC' to exit")
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done:
            # Render the environment
            env.render()
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("Quit requested by user")
                env.close()
                return
            
            # Select action using trained policy
            action = agent.select_action(np.array(state))
            continuous_action = discrete_to_continuous_action(action)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(continuous_action)
            episode_reward += reward
            step_count += 1
            
            state = next_state
            done = terminated or truncated
            
            # Optional: Add some delay to see the action
            time.sleep(0.01)
            
            # Break if episode is too long or reward is very negative
            if step_count > 1000 or episode_reward < -100:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} finished: Reward = {episode_reward:.2f}, Steps = {step_count}")
    
    env.close()
    
    # Print statistics
    avg_reward = np.mean(total_rewards)
    max_reward = np.max(total_rewards)
    min_reward = np.min(total_rewards)
    
    print(f"\nTest Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")
    print(f"Min Reward: {min_reward:.2f}")
    print(f"All Rewards: {total_rewards}")

if __name__ == "__main__":
    # Test with the default trained agent
    test_agent("trained_agent.pth", num_episodes=3)
    
    # Alternative: Test with a specific checkpoint
    # test_agent("checkpoint_episode_2000.pth", num_episodes=3)
