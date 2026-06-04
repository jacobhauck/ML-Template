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


class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


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


class StackedLinear(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_units: int,
            bias: bool = True,
            device=None,
            dtype=None
    ):
        """
        A stack of linear layers
        :param in_features: Number of input features
        :param out_features: Number of output features
        :param num_units: Number of individual linear layers
        :param bias: Whether to use a learnable bias
        :param device: Optional device on which to allocate the parameters
        :param dtype: Optional data type with which tot allocate the parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_units = num_units

        kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(
            torch.empty((num_units, in_features, out_features), **kwargs)
        )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty((num_units, out_features), **kwargs)
            )
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / self.in_features**.5
        torch.nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        :param x: (..., num_units, in_features) input data
        :return: (..., num_units, out_features) output data
        """

        x = torch.einsum('...ui,uio->...uo', x, self.weight)
        # (..., num_units, out_features)

        if self.bias is not None:
            x = x + self.bias
            # (..., num_units, out_features)

        return x

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class StackedMLP(torch.nn.Module):
    def __init__(
            self,
            d_in: int,
            hidden_layers: Sequence[int],
            d_out: int,
            num_units: int,
            activation: Sequence[Mapping] | Mapping,
            bias: Sequence[bool] | bool = True,
            norm: Sequence[Mapping] | Mapping | None = None,
            dropout: Sequence[float] | float | None = None,
            order: str = 'nad'
    ):
        """
        A stack of multi-layer perceptrons applied separately.
        :param d_in: input dimension
        :param hidden_layers: sequence of integers giving hidden dimensions
            (for each unit in the stack, not the sum of all units); number of
            layers = len(hidden_layers) + 1
        :param d_out: output dimension
        :param num_units: number of individual units in the stack
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
        super().__init__()

        # Save parameters
        self.d_in = d_in
        self.d_out = d_out
        self.hidden_layers = tuple(hidden_layers)
        self.num_units = num_units
        self.order = order

        # Construct layers
        layers = []
        self.activation = _make_sequence(activation, len(self.hidden_layers))
        self.bias = _make_sequence(bias, len(self.hidden_layers) + 1)
        self.norm = _make_sequence(norm, len(self.hidden_layers))
        self.dropout = _make_sequence(dropout, len(self.hidden_layers))

        architecture = (d_in,) + self.hidden_layers + (d_out,)
        for i, (d_in, d_out) in enumerate(zip(architecture[:-1], architecture[1:])):
            if i == 0:
                # First layer is regular linear; this duplicates the input for each group
                layers.append(torch.nn.Linear(d_in, d_out * num_units, bias=self.bias[i]))
                # Manually unflatten the last dimension for use with StackedLinear
                layers.append(torch.nn.Unflatten(-1, (num_units, d_out)))
            else:
                layers.append(StackedLinear(d_in, d_out, num_units, bias=self.bias[i]))

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
        """
        :param x: (..., d_in)
        :return: (..., num_units, d_out)
        """
        return self.layers(x)  # (..., num_units, d_out)


class RelativeL2Loss(torch.nn.Module):
    def __init__(self, squared=True):
        super().__init__()
        self.squared = squared
    
    def forward(self, x, target):
        """
        Calculates relative L^2 loss
        :param x: (B, *shape, d_out)
        :param target: (B, *shape, d_out)
        :return: Mean relative L^2 loss across the given batch
        """
        # We can take the mean over both spatial shape and d_out because the
        # extraneous factor of 1/d_out (or sqrt(1/d_out)) cancels when we divide
        # by size, which has the same factor
        abs_loss = ((x - target)**2).reshape(x.shape[0], -1).mean(dim=1)
        size = (target**2).reshape(target.shape[0], -1).mean(dim=1)
        # (B,) each
        
        if self.squared:
            return (abs_loss / size).mean()
        else:
            return ((abs_loss / size) ** .5).mean()

    def __repr__(self):
        return f'RelativeL2Loss(squared={self.squared})'
