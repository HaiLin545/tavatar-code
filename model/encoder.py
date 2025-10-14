import torch
from torch import nn
import tinycudann as tcnn

class HashEncoder(nn.Module):
    def __init__(self, num_channels, num_players=1, n_input_dims=3):
        super().__init__()
        self.networks = []
        self.num_players = num_players
        for i in range(num_players):
            self.networks.append(
                tcnn.NetworkWithInputEncoding(
                    n_input_dims=n_input_dims,
                    n_output_dims=num_channels,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": 16,
                        "n_features_per_level": 4,
                        "log2_hashmap_size": 17,
                        "base_resolution": 4,
                        "per_level_scale": 1.5,
                    },
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 64,
                        "n_hidden_layers": 2,
                    },
                )
            )
        self.networks = nn.ModuleList(self.networks)

    def forward(self, x):
        self.outputs = []
        for i in range(self.num_players):
            self.outputs.append(self.networks[i](x[i]))
        return torch.stack(self.outputs).float()

class DisplacementEncoder(nn.Module):
    def __init__(self, num_players=1):
        super().__init__()
        self.num_players = num_players
        self.input_channels = 3
        self.encoder = HashEncoder(3, num_players=num_players)

    def forward(self, input):
        if input.shape[0] != self.num_players or input.shape[2] != self.input_channels:
            raise Exception("input shape is not allowed")
        return self.encoder(input)
