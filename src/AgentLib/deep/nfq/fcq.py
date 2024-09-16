# AgentLib
# 2024 Braedan Kennedy (kennedyengineering)

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCQ(nn.Module):
    """
    Fully connected Q-function (state-in-values-out)
    """

    def __init__(
        self, input_dim, output_dim, hidden_dims=(32, 32), activation_fc=F.relu
    ):
        """
        Initialization method
        """

        # Initialize nn.Module
        super().__init__()

        # Store activation function
        self.activation_fc = activation_fc

        # Define input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        # Define output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Define and set device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        self.device = torch.device(device)
        self.to(device)

    def _format(self, state):
        """
        Convert the raw state to a tensor
        """

        x = state

        # Check datatype
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        return x

    def forward(self, state):
        """
        Forward pass through the neural network
        """

        # Convert the raw state to a tensor
        x = self._format(state)

        # Pass the state through the input layer and activation function
        x = self.activation_fc(self.input_layer(x))

        # Pass the state through the hidden layers and activation functions
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))

        # Pass the state through the output layer (and NOT the activation function)
        x = self.output_layer(x)

        return x
