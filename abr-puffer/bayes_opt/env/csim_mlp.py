from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, activation):
        super().__init__()
        sizes = [input_dim] + list(hidden_sizes) + [output_dim]
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else nn.Identity
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        self.predict = nn.Sequential(*layers)

    def forward(self, raw_input):
        prediction = self.predict(raw_input)
        return prediction
