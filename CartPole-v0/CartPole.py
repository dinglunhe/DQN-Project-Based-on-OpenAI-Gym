import gym
from DQN import DeepQNetwork

env = gym.make('CartPole-v0')  # Define which environment in the gym library to use
env = env.unwrapped

print(env.action_space)  # See how many actions are available in this environment
print(env.observation_space)  # See how many observations in this environment
print(env.observation_space.shape)
print(env.observation_space.high)  # View observation highest value
print(env.observation_space.low)  # View observation lowest value
print(env.reward_range)  # View the range of reward

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001)

total_steps = 0
total_reward_list = []
for i_episode in range(100):
    # Initialize the environment
    observation = env.reset()
    total_reward = 0
    while True:
        # Refresh environment
        env.render()
        # DQN select behavior based on observations
        action = RL.choose_action(observation)
        # Give the next state and reward according to the action
        observation_, reward, done, info = env.step(action)

        # x is the horizontal displacement of the car,
        # so r1 is the more off-center the car is, the lower the score

        # theta is the vertical angle of the stick, the larger the angle, the less vertical the stick,
        # so r2 is the more vertical the stick, the higher the score

        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5

        # Total reward is a combination of r1 and r2, considering both position and angle,
        # so that DQN learning is more efficient
        reward = r1 + r2

        # Store memory
        RL.store_transition(observation, action, reward, observation_)
        # Total rewards within an episode
        total_reward += reward

        # Control learning start time
        if total_steps > 1:
            RL.learn()

        # If terminated, exit the loop
        if done:
            total_reward_list.append(round(total_reward, 2))
            print('episode: ', i_episode,
                  'total_reward: ', round(total_reward, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            print('total reward list:', total_reward_list)
            break

        # Change state_ to the state of the next loop
        observation = observation_
        # Total steps
        total_steps += 1
