import mlx
import torch
import matplotlib.pyplot as plt


@mlx.experiment
def verify_dist(config, name, group=None):
    model = mlx.create_module(config['model'])
    x = torch.rand((config['n'], model.d_in)) * 2 - 1

    plt.hist(x.flatten(), bins=50, density=True)
    plt.show()
    for layer in model.layers:
        x = layer(x)
        plt.hist(x.flatten().detach(), bins=50, density=True)
        plt.title(repr(layer))
        plt.show()

    print(x.abs().mean())