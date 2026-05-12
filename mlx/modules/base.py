import importlib
import os

import mlx.modules
from typing import Mapping


debug_search_paths = False


def _module_file_path():
    folder = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(folder, 'path.txt')


def add_module_path(path):
    module_file = _module_file_path()
    if os.path.exists(module_file):
        with open(module_file) as f:
            existing_paths = f.read().splitlines()
    else:
        with open(module_file, 'w') as f:
            existing_paths = ()
    
    if path not in existing_paths:
        with open(_module_file_path(), 'a') as f:
            f.write(path + '\n')


def remove_module_path(path):
    module_file = _module_file_path()
    if os.path.exists(module_file):
        with open(module_file) as f:
            existing_paths = f.read().splitlines()
    else:
        existing_paths = ()
    
    if path in existing_patshs:
        with open(_module_file_path(), 'w') as f:
            for other_path in existing_paths:
                if other_path != path:
                    f.write(other_path + '\n')


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
        - the Module's name as a global import,
        - the Module's name relative to a package added via `add_module_path`,
        - or the Module's name as an import relative to the `modules` package.
    :return: An instance of the Module specified by the given configuration
        settings.
    """
    config = dict(config)
    name_str = config.pop('name')

    try:
        # Standard torch module
        import torch
        return getattr(torch.nn, name_str)(**config)
    except AttributeError:
        if debug_search_paths:
            print(f'Module {name_str} not found as torch module')

    try:
        # mlx built-in module
        return getattr(mlx.modules, name_str)(**config)
    except AttributeError:
        if debug_search_paths:
            print(f'Module {name_str} not found as mlx module')

    # Global import
    try:
        name = name_str.split('.')
        py_module = importlib.import_module('.'.join(name[:-1]))
        return getattr(py_module, name[-1])(**config)
    except (ModuleNotFoundError, ValueError):
        if debug_search_paths:
            print(f'Module {name_str} not found as global import')

    # User-defined module in modules
    try:
        name = ['modules'] + name_str.split('.')
        py_module = importlib.import_module('.'.join(name[:-1]))
        return getattr(py_module, name[-1])(**config)
    except (ModuleNotFoundError, AttributeError):
        if debug_search_paths:
            print(f'Module {name_str} not found in local `modules` folder')

    # User-defined module in added module path
    added_path = _module_file_path()
    if not os.path.exists(added_path):
        raise ValueError('Invalid Module')
    
    with open(added_path) as f:
        paths = f.read().splitlines()
    
    for path in paths:
        try:
            name = path.split('.') + name_str.split('.')
            py_module = importlib.import_module('.'.join(name[:-1]))
            return getattr(py_module, name[-1])(**config)
        except ModuleNotFoundError:
            if debug_search_paths:
                print(f'Module {name_str} not found in custom path: {path}')

    raise ValueError('Invalid Module')
