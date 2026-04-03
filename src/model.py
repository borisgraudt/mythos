import torch
import torch.nn as nn

from src.utils import get_device


class Mythos(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = get_device()

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.output = nn.Linear(config['d_model'], config['vocab_size'])

        self.layers = nn.ModuleList([
            nn.Linear(config['d_model'], config['d_model'])
            for _ in range(config['n_layers'])
        ])

        self.to(self.device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return x


    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
