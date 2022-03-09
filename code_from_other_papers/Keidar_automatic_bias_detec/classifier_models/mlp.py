import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, dimensions, device, dropout=0, activation=nn.ReLU):
        """

        Parameters
        ----------
        dimensions
        dropout (int): hardcoded to 0
        activation: hardcoded to nn.ReLU
        device: passed
        """
        super().__init__()

        self.device = device

        # Is it binary classification or not?
        self.binary = (dimensions[-1] == 1)
        self.layers = nn.ModuleList()

        # create the structure of the MLP with the respective dimensions
        for in_dim, out_dim in zip(dimensions[:-2], dimensions[1:-1]):
            # loop through dimension list, add activation function after each Linear
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    activation()
                )
            )
            if dropout:
                self.layers.append(nn.Dropout(dropout))

        # add the layer from the previous layers to the last classification layer
        self.layers.append(
            nn.Linear(dimensions[-2], dimensions[-1])
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.binary:
            x = x.squeeze(1)
        return x

    def predict(self, x):
        if self.binary:
            return torch.round(torch.sigmoid(self.forward(x)))
        return self.forward(x)