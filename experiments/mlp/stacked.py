import mlx
import json
import torch

@mlx.experiment
def run_experiment(config, name, group=None):
    configs = [
        {
            'd_in': 1,
            'hidden_layers': [3, 4, 5, 6],
            'd_out': 1,
            'num_units': 2,
            'activation': {'name': 'ReLU'}
        },
        {
            'd_in': 2,
            'hidden_layers': [8, 8, 8],
            'd_out': 2,
            'num_units': 2,
            'activation': [
                {'name': 'ReLU'},
                {'name': 'SELU'},
                {'name': 'Sigmoid'}
            ]
        },
        {
            'd_in': 3,
            'hidden_layers': [5, 5, 5],
            'd_out': 3,
            'num_units': 2,
            'activation': {'name': 'ReLU'},
            'norm': {'name': 'LayerNorm', 'normalized_shape': 5}
        },
        {
            'd_in': 4,
            'hidden_layers': [5, 6, 7],
            'd_out': 1,
            'num_units': 2,
            'activation': {'name': 'Tanh'},
            'norm': [
                {'name': 'LayerNorm', 'normalized_shape': 5},
                {'name': 'LayerNorm', 'normalized_shape': 6},
                {'name': 'LayerNorm', 'normalized_shape': 7}
            ]
        },
        {
            'd_in': 5,
            'hidden_layers': [5, 6, 7],
            'd_out': 1,
            'num_units': 2,
            'activation': {'name': 'Tanh'},
            'norm': [
                {'name': 'LayerNorm', 'normalized_shape': 5},
                {'name': 'LayerNorm', 'normalized_shape': 6},
                {'name': 'LayerNorm', 'normalized_shape': 7}
            ],
            'dropout': 0.3
        },
        {
            'd_in': 6,
            'hidden_layers': [5, 6, 7],
            'd_out': 1,
            'num_units': 2,
            'activation': [
                {'name': 'Tanh'},
                {'name': 'ReLU'},
                {'name': 'LeakyReLU', 'negative_slope': 0.1}
            ],
            'norm': [
                {'name': 'LayerNorm', 'normalized_shape': 5},
                {'name': 'LayerNorm', 'normalized_shape': 6},
                {'name': 'LayerNorm', 'normalized_shape': 7}
            ],
            'dropout': [0.1, 0.2, 0.3]
        },
        {
            'd_in': 7,
            'hidden_layers': [5, 6, 7],
            'd_out': 1,
            'num_units': 2,
            'activation': [
                {'name': 'Tanh'},
                {'name': 'ReLU'},
                {'name': 'LeakyReLU', 'negative_slope': 0.1}
            ],
            'norm': [
                {'name': 'LayerNorm', 'normalized_shape': 5},
                {'name': 'LayerNorm', 'normalized_shape': 6},
                {'name': 'LayerNorm', 'normalized_shape': 7}
            ],
            'dropout': [0.1, 0.2, 0.3],
            'order': 'nda'
        },
        {
            'd_in': 7,
            'hidden_layers': [5, 6, 7],
            'd_out': 1,
            'num_units': 2,
            'activation': [
                {'name': 'Tanh'},
                {'name': 'ReLU'},
                {'name': 'LeakyReLU', 'negative_slope': 0.1}
            ],
            'norm': [
                {'name': 'LayerNorm', 'normalized_shape': 5},
                {'name': 'LayerNorm', 'normalized_shape': 6},
                {'name': 'LayerNorm', 'normalized_shape': 7}
            ],
            'dropout': [0.1, 0.2, 0.3],
            'order': 'adn'
        },
        {
            'd_in': 8,
            'hidden_layers': [5, 6, 7],
            'd_out': 1,
            'num_units': 2,
            'activation': [
                {'name': 'Tanh'},
                {'name': 'ReLU'},
                {'name': 'LeakyReLU', 'negative_slope': 0.1}
            ],
            'dropout': [0.1, 0.2, 0.3],
            'order': 'dna'
        },
        {
            'd_in': 9,
            'hidden_layers': [5, 6, 7],
            'd_out': 1,
            'num_units': 2,
            'activation': [
                {'name': 'Tanh'},
                {'name': 'ReLU'},
                {'name': 'LeakyReLU', 'negative_slope': 0.1}
            ],
            'norm': [
                {'name': 'LayerNorm', 'normalized_shape': 5},
                {'name': 'LayerNorm', 'normalized_shape': 6},
                {'name': 'LayerNorm', 'normalized_shape': 7}
            ],
            'order': 'and'
        }
    ]

    for c in configs:
        print(json.dumps(c, indent=2))
        print()

        model = mlx.modules.StackedMLP(**c)
        print(model)
        print()

        x = torch.rand((3, 5, 2, model.d_in))
        print(x.shape, model(x).shape)
        print()
