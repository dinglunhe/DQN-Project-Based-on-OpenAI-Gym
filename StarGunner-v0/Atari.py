import cv2
import gym
from DQN import DeepQNetwork

env = gym.make('StarGunner-v0')  # Define which environment in the gym library to use
env = env.unwrapped

print(env.action_space)  # See how many actions are available in this environment
print(env.observation_space)  # See how many observations in this environment
print(env.observation_space.shape)
print(env.observation_space.high)  # View observation highest value
print(env.observation_space.low)  # View observation lowest value
print(env.reward_range)  # View the range of reward

inputImageSize = (100, 80, 1)

RL = DeepQNetwork(actions=env.action_space.n,
                  observation_shape=inputImageSize,
                  learning_rate=0.01, epsilon_max=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.0001)

total_steps = 0
total_reward_list = []
for i_episode in range(2000):
    # Initialize the environment
    observation = env.reset()
    # Use opencv for graying
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    observation = cv2.resize(observation, (inputImageSize[1], inputImageSize[0]))  # cv2.resize(img, (width, height))
    total_reward = 0
    while True:
        # Refresh environment
        env.render()
        # DQN select behavior based on observations
        action = RL.choose_action(observation)
        # Give the next state and reward according to the action
        observation_, reward, done, info = env.step(action)
        # Use opencv for graying
        observation_ = cv2.cvtColor(observation_, cv2.COLOR_BGR2GRAY)
        observation_ = cv2.resize(observation_, (inputImageSize[1], inputImageSize[0]))
        # Store memory
        RL.store_transition(observation, action, reward, observation_)
        # Total rewards within an episode
        total_reward += reward

        # Control learning start time
        if total_steps > 1000:
            RL.learn()

        # If terminated, exit the loop
        if done:
            total_reward_list.append(total_reward)
            print('episode: ', i_episode,
                  'total_reward: ', round(total_reward, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            print('total reward list:', total_reward_list)
            break

        # Change state_ to the state of the next loop
        observation = observation_
        # Total steps
        total_steps += 1
