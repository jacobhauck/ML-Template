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
        self.input_names.add(key)

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


def _make_sequence(data, size):
    try:
        _ = data[0]
        return data
    except (TypeError, KeyError):  # KeyError in case data is a config dict
        return [data] * size


class MLP(torch.nn.Module):
    def __init__(
            self,
            d_in: int,
            hidden_layers: Sequence[int],
            d_out: int,
            activation: Sequence[Mapping] | Mapping,
            bias: Sequence[bool] | bool = True,
            norm: Sequence[Mapping] | Mapping | None = None,
            dropout: Sequence[float] | float | None = None,
            order: str = 'nad'
    ):
        """
        A multi-layer perceptron.
        :param d_in: input dimension
        :param hidden_layers: sequence of integers giving hidden dimensions;
            number of layers = len(hidden_layers) + 1
        :param d_out: output dimension
        :param bias: Whether to use bias in the linear layers. Either a list of
            per-layer values or one to apply to all layers. Default = True.
        :param activation: Activation function config or per-layer list of
            configs to apply in hidden layers (list of size len(hidden_layers)).
        :param norm: Normalization config to apply for each hidden layer; either
            a layer-by-layer list (of size len(hidden_layers)), or one value to
            apply to all layers, or None for no normalization. Default = None.
        :param dropout: Dropout rate for dropout applied after each hidden layer;
            either a layer-by-layer list (of size len(hidden_layers)), or one
            value to apply to all layers. None for no dropout. Default = None.
        :param order: Order in which to apply activation, dropout and
            normalization (if the latter two are specified) as a string of the
            characters 'a', 'd', 'n'. Default = 'nad', that is,
            normalization, activation, dropout
        """
        super(MLP, self).__init__()

        # Save parameters
        self.d_in = d_in
        self.d_out = d_out
        self.hidden_layers = tuple(hidden_layers)
        self.order = order

        # Construct layers
        layers = []
        self.activation = _make_sequence(activation, len(self.hidden_layers))
        self.bias = _make_sequence(bias, len(self.hidden_layers) + 1)
        self.norm = _make_sequence(norm, len(self.hidden_layers))
        self.dropout = _make_sequence(dropout, len(self.hidden_layers))

        architecture = (d_in,) + self.hidden_layers + (d_out,)
        for i, (d_in, d_out) in enumerate(zip(architecture[:-1], architecture[1:])):
            layers.append(torch.nn.Linear(d_in, d_out, bias=self.bias[i]))
            if i < len(self.hidden_layers):
                post_layer_modules = {'a': mlx.create_module(self.activation[i])}

                if self.norm[i] is not None:
                    post_layer_modules['n'] = mlx.create_module(self.norm[i])

                if self.dropout[i] is not None:
                    post_layer_modules['d'] = torch.nn.Dropout(self.dropout[i])

                for module in order:
                    if module in post_layer_modules:
                        layers.append(post_layer_modules[module])

        # Save layers as Sequential module
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class RelativeL2Loss(torch.nn.Module):
    def __init__(self, squared=True):
        super().__init__()
        self.squared = squared
    
    def forward(self, x, target):
        abs_loss = ((x - target)**2).view(x.shape[0], -1).mean(dim=1)
        size = (target**2).view(target.shape[0], -1).mean(dim=1)
        
        if self.squared:
            return (abs_loss / size).mean()
        else:
            return ((abs_loss / size) ** .5).mean()
