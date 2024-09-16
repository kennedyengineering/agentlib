# AgentLib
# 2024 Braedan Kennedy (kennedyengineering)

import torch
import numpy as np


class EGreedyStrategy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.exploratory_action_taken = None

    def select_action(self, model, state):
        self.exploratory_action_taken = False

        with torch.no_grad():
            # Extract the Q-values for state s
            q_values = model(state).cpu().detach()

            # Make the values NumPY friendly
            q_values = q_values.data.numpy().squeeze()

        if np.random.rand() > self.epsilon:
            # Act greedily if random number is greater than epsilon
            action = np.argmax(q_values)
        else:
            # Act randomly if random number is less than epsilon
            action = np.random.randint(len(q_values))

        self.exploratory_action_taken = action != np.argmax(q_values)

        return action
