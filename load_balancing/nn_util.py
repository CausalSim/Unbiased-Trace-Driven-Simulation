
import torch.nn as nn

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, activation):
        super().__init__()
        self.predict = mlp(
            sizes=[input_dim] + list(hidden_sizes) + [output_dim],
            activation=activation,
            output_activation=nn.Identity,
        )

    def forward(self, raw_input):
        prediction = self.predict(raw_input)
        return prediction
