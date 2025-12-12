from typing import Mapping

import torch

"""
Dictionary of custom, named optimizers. If an optimizer is requested in
create_optimizer that is not present in the torch.optim namespace, then create_optimizer will
look for the module in this dictionary.
"""
custom_optimizers = {  # dict[str, Optimizer]

}


def create_optimizer(parameters, config: Mapping):
    """
    Create an optimizer from its configuration.

    :param parameters: model parameters to assign to the optimizer
    :param config: Dictionary of optimizer configuration options. Must contain a
        'name' field to specify which optimizer to load, which refers to the
        name of the optimizer within the torch.optim module, or else the name
        used as a key in the optim.custom_optimizers dictionary of
        custom optimizers.
    :return: An instance of the optimizer specified by the given
        configuration settings.
    """
    import torch

    config = dict(config)
    name = config.pop('name')

    module = getattr(torch.optim, name, None)
    if module is None:
        module = custom_optimizers[name]

    return module(parameters, **config)


# noinspection PyProtectedMember
def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for sub_param in param.values():
                if isinstance(sub_param, torch.Tensor):
                    sub_param.data = sub_param.data.to(device)
                    if sub_param._grad is not None:
                        sub_param._grad.data = sub_param._grad.data.to(device)

    return optim
