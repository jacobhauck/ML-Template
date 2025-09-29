import importlib
import torch
import mlx.modules
from typing import Mapping


def create_model(config: Mapping):
    """
    Alias of create_module for compatibility
    """
    return create_module(config)



def create_module(config: Mapping):
    """
    Create a Module from its configuration.

    :param config: Dictionary of Module configuration options. Must contain a
        'name' field to specify which Module to load, which refers to one of the
        following:
        - the Module's name in torch.nn,
        - the Module's name in mlx.modules,
        - or the Module's name within the project `modules` directory.
    :return: An instance of the Module specified by the given configuration
        settings.
    """
    config = dict(config)
    name_str = config.pop('name')

    try:
        # Standard torch module
        return getattr(torch.nn, name_str)
    except AttributeError:
        pass

    try:
        # mlx built-in module
        return getattr(mlx.modules, name_str)
    except AttributeError:
        pass

    # Otherwise look for user-defined module
    name = ['modules'] + name_str.split('.')
    py_module = importlib.import_module('.'.join(name[:-1]))

    return getattr(py_module, name[-1])(**config)
