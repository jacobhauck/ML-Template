from typing import Sequence, Mapping

import torch

import mlx


class BufferDict(torch.nn.Module):
    def __init__(self, input_dict):
        super().__init__()
        self.input_names = set(input_dict)
        for k, v in input_dict.items():
            self.register_buffer(k, v)

    def __len__(self):
        return len(self.input_names)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, item):
        return item in self.input_names

    def items(self):
        for name in self.input_names:
            yield name, getattr(self, name)

    def keys(self):
        yield from self.input_names

    def values(self):
        for name in self.input_names:
            yield getattr(self, name)


class MLP(torch.nn.Module):
    def __init__(
            self,
            d_in: int,
            hidden_layers: Sequence[int],
            d_out: int,
            activation: Mapping,
            bias: bool = True
    ):
        """
        A multi-layer perceptron.
        :param d_in: input dimension
        :param hidden_layers: sequence of integers giving hidden dimensions;
            number of layers = len(hidden_layers) + 1
        :parma d_out: output dimension
        :param activation: Activation function config
        :param bias: Whether to use bias in the linear layers. Default=True.
        """
        super(MLP, self).__init__()

        # Save parameters
        self.d_in = d_in
        self.d_out = d_out
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation

        # Construct layers
        layers = []

        architecture = (d_in,) + self.hidden_layers + (d_out,)
        for d_in, d_out in zip(architecture[:-1], architecture[1:]):
            layers.append(torch.nn.Linear(d_in, d_out, bias=bias))
            layers.append(mlx.create_module(activation))
        layers.pop()  # Remove the last activation function

        # Save layers as Sequential module
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class RelativeL2Loss(torch.nn.Module):
    def forward(self, x, target):
        abs_loss = ((x - target)**2).view(x.shape[0], -1).mean(dim=1)
        size = (target**2).view(target.shape[0], -1).mean(dim=1)
        return (abs_loss / size).mean()
