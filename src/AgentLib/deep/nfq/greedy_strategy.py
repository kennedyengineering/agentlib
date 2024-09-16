# AgentLib
# 2024 Braedan Kennedy (kennedyengineering)

import torch
import numpy as np


class GreedyStrategy:
    def __init__(self):
        self.exploratory_action_taken = False

    def select_action(self, model, state):

        with torch.no_grad():
            # Extract the Q-values for state s
            q_values = model(state).cpu().detach()

            # Make the values NumPY friendly
            q_values = q_values.data.numpy().squeeze()
