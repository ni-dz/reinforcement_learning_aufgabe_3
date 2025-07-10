# utils.py
import matplotlib.pyplot as plt
import numpy as np

def setup_live_plot():
    plt.ion()
    fig, ax = plt.subplots()
    rewards_list, losses_list, epsilon_list = [], [], []
    reward_line, = ax.plot(rewards_list, label='Reward')
    loss_line, = ax.plot(losses_list, label='Loss')
    epsilon_line, = ax.plot(epsilon_list, label='Epsilon')
    ax.legend()
    plt.show()
    return fig, ax, reward_line, loss_line, epsilon_line, rewards_list, losses_list, epsilon_list

def update_plot(rewards_list, losses_list, epsilon_list, reward_line, loss_line, epsilon_line, ax):
    reward_line.set_ydata(rewards_list)
    reward_line.set_xdata(range(len(rewards_list)))
    loss_line.set_ydata(losses_list)
    loss_line.set_xdata(range(len(losses_list)))
    epsilon_line.set_ydata(epsilon_list)
    epsilon_line.set_xdata(range(len(epsilon_list)))
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

def moving_average(data, window_size=10):
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
