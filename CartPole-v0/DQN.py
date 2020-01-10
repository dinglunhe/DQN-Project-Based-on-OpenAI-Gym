# -*- coding: UTF-8 -*-

import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop


# Deep Q Network
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # Record learning steps (used to determine whether to change the target_net parameter)
        self.learn_step_counter = 0

        # Initialize memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # Create target_net and evaluate_net
        self._build_net()

    def _build_net(self):
        # ------------------ Build evaluate_net ------------------
        self.model_eval = Sequential([
            # Input
            Dense(64, input_dim=self.n_features),
            Activation('relu'),
            # Output
            Dense(self.n_actions),
        ])
        # Select the RMS optimizer and choose the learning rate parameters
        rmsprop = RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model_eval.compile(loss='mse',
                                optimizer=rmsprop,
                                metrics=['accuracy'])

        # ------------------ Build target_net ------------------
        # The architecture of the target_net must be the same as the evaluate_net,
        # but it does not need to calculate the loss function
        self.model_target = Sequential([
            # Input
            Dense(64, input_dim=self.n_features),
            Activation('relu'),
            # Output
            Dense(self.n_actions),
        ])

    def store_transition(self, s, a, r, s_):
        # # The hasattr() is used to determine whether an object contains the corresponding attributes.
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        # Replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        # The total memory size is fixed.
        # If the total memory size is exceeded, the old memory is replaced by the new memory.
        self.memory[index, :] = transition  # Replace

        self.memory_counter += 1

    def choose_action(self, observation):
        # Insert a new dimension, which is needed for matrix calculation
        # To have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # Let the eval_net generate the values of all actions,
            # and select the action with the highest value
            actions_value = self.model_eval.predict(observation)
            action = np.argmax(actions_value)
        else:
            # Randomly select
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # Parameter replacement after a certain number of steps
        # Check whether to replace the target_net parameter
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.model_target.set_weights(self.model_eval.get_weights())
            print('\ntarget_params_replaced\n')

        # Randomly extract batch_size amount of memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # Get q_next (q generated by target_net) and q_eval (q generated by eval_net)
        q_next = self.model_target.predict(batch_memory[:, -self.n_features:], batch_size=self.batch_size)
        q_eval = self.model_eval.predict(batch_memory[:, :self.n_features], batch_size=self.batch_size)

        # Change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # Training eval_net
        self.cost = self.model_eval.train_on_batch(batch_memory[:, :self.n_features], q_target)

        # Increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
