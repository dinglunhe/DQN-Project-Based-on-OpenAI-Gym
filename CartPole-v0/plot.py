import numpy as np
import matplotlib.pyplot as plt


def plot_reward(total_reward_list):
    plt.figure()
    plt.subplot(121)
    plt.plot(np.arange(len(total_reward_list)), total_reward_list)
    plt.ylabel('Reward')
    plt.xlabel('Training Episode')

    plt.subplot(122)
    plt.plot(np.convolve(total_reward_list, np.ones(10) / 10, mode='valid'))
    plt.ylabel('Moving Average of Reward (per 10 Episodes)')
    plt.xlabel('Training Episode')

    plt.show()


reward = [1.72, 4.21, 1.52, 9.81, 1.87, 4.39, 2.45, 11.53, 22.31, 5.92, -0.15, 4.43, 7.27, 17.28, 2.55, 14.94, 10.74, 10.43, 48.29, 42.13, 95.56, 96.18, 88.17, 44.06, 36.03, 85.8, 114.83, 52.59, 41.23, 49.45, 60.84, 167.26, 156.5, 113.58, 127.05, 33.3, 137.65, 301.22, 304.6, 99.64, 69.29, 164.23, 171.97, 271.65, 310.16, 131.62, 166.88, 118.41, 157.68, 120.79, 429.09, 65.74, 247.61, 154.75, 136.11, 123.11, 863.26, 388.93, 202.83, 139.21, 96.95, 59.85, 222.09, 427.12, 127.76, 210.89, 562.21, 176.25, 123.32, 101.16, 139.37, 249.14, 324.02, 185.04, 788.96, 209.36, 215.97, 771.32, 228.64, 735.55, 144.96, 713.3, 249.52, 193.35, 190.24, 256.03, 573.8, 249.88, 174.51, 277.78, 118.59, 1027.8, 322.9, 196.4, 342.2, 473.04, 292.53, 239.21, 1085.67, 569.75]

plot_reward(reward)
