# -*- coding: UTF-8 -*-


import numpy as np
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop


# Deep Q Network
class DeepQNetwork:
    def __init__(
            self,
            actions,
            observation_shape,
            learning_rate=0.01,
            reward_decay=0.9,
            epsilon_max=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None
    ):
        self.actions = actions
        self.observation_shape = observation_shape
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = epsilon_max
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # Record learning steps (used to determine whether to change the target_net parameter)
        self.learn_step_counter = 0

        # Initialize memory [s, a, r, s_]
        # Since the image data is too large, separate and save it with numpy
        self.memoryObservationNow = np.zeros((self.memory_size, self.observation_shape[0],
                                              self.observation_shape[1], self.observation_shape[2]), dtype='int16')
        self.memoryObservationLast = np.zeros((self.memory_size, self.observation_shape[0],
                                               self.observation_shape[1], self.observation_shape[2]), dtype='int16')
        self.memoryReward = np.zeros(self.memory_size, dtype='float64')
        self.memoryAction = np.zeros(self.memory_size, dtype='int16')

        # Create target_net and evaluate_net
        self._build_net()

    def _build_net(self):
        # ------------------ Build evaluate_net ------------------
        self.model_eval = Sequential([
            # The input first layer is a 2D convolutional layer (100, 80, 1)
            Convolution2D(  # Conv2D layer
                batch_input_shape=(None, self.observation_shape[0], self.observation_shape[1],
                                   self.observation_shape[2]),
                filters=15,  # Number of convolution kernels
                kernel_size=5,  # Convolution kernel width and length
                strides=1,  # Swipe size
                padding='same',  # The size of the filtered data is the same as before
                data_format='channels_last',  # The last dimension of the rgb image represents the channel
            ),
            # output(100, 80, 15)
            Activation('relu'),
            # Pooling layer output shape (50, 40, 15)
            MaxPooling2D(
                pool_size=2,
                strides=2,
                padding='same',
                data_format='channels_last',
            ),
            # output(50, 40, 30)
            Convolution2D(30, 5, strides=1, padding='same', data_format='channels_last'),
            Activation('relu'),
            # output(10, 8, 30)
            MaxPooling2D(5, 5, 'same', data_format='channels_first'),
            Flatten(),
            # output(512)
            Dense(512),
            Activation('relu'),
            # output(18)
            Dense(self.actions),
        ])
        # Choose RMS optimizer
        rmsprop = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model_eval.compile(loss='mse',
                                optimizer=rmsprop,
                                metrics=['accuracy'])

        # ------------------ Build target_net ------------------
        # The architecture of the target_net must be the same as the evaluate_net,
        # but it does not need to calculate the loss function
        self.model_target = Sequential([
            Convolution2D(  # Conv2D layer
                batch_input_shape=(None, self.observation_shape[0], self.observation_shape[1],
                                   self.observation_shape[2]),
                filters=15,
                kernel_size=5,
                strides=1,
                padding='same',
                data_format='channels_last',
            ),
            # output(100, 80, 15)
            Activation('relu'),
            # Pooling layer output shape (50, 40, 15)
            MaxPooling2D(
                pool_size=2,
                strides=2,
                padding='same',
                data_format='channels_last',
            ),
            # output(50, 40, 30)
            Convolution2D(30, 5, strides=1, padding='same', data_format='channels_last'),
            Activation('relu'),
            # output(10, 8, 30)
            MaxPooling2D(5, 5, 'same', data_format='channels_first'),
            Flatten(),
            # output(512)
            Dense(512),
            Activation('relu'),
            # output(18)
            Dense(self.actions),
        ])

    def store_transition(self, s, a, r, s_):
        # The hasattr() is used to determine whether an object contains the corresponding attributes.
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        s = s[:, :, np.newaxis]
        s_ = s_[:, :, np.newaxis]
        # Replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        # The total memory size is fixed.
        # If the total memory size is exceeded, the old memory is replaced by the new memory.
        self.memoryObservationNow[index, :] = s_
        self.memoryObservationLast[index, :] = s
        self.memoryReward[index] = r
        self.memoryAction[index] = a

        self.memory_counter += 1

    def choose_action(self, observation):
        # Insert new dimensions, which are needed for matrix calculation
        # To have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :, :, np.newaxis]

        if np.random.uniform() < self.epsilon:
            # Let the eval_net generate the values of all actions,
            # and select the action with the highest value
            actions_value = self.model_eval.predict(observation)
            action = np.argmax(actions_value)
        else:
            # Randomly select
            action = np.random.randint(0, self.actions)
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

        batch_memoryONow = self.memoryObservationNow[sample_index, :]
        batch_memoryOLast = self.memoryObservationLast[sample_index, :]
        batch_memoryAction = self.memoryAction[sample_index]
        batch_memoryReward = self.memoryReward[sample_index]

        # Get q_next (q generated by target_net) and q_eval (q generated by eval_net)
        q_next = self.model_target.predict(batch_memoryONow, batch_size=self.batch_size)
        q_eval = self.model_eval.predict(batch_memoryOLast, batch_size=self.batch_size)

        # Change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memoryAction.astype(int)
        reward = batch_memoryReward

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # Training eval_net
        self.cost = self.model_eval.train_on_batch(batch_memoryONow, q_target)

        # Increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
