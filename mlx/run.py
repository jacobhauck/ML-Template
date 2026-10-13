"""
Script to run an experiment or a group of experiments.

Usage: python -m run_experiment <name> [<config>...] [--group | -g <group>]

Functions
---------
    - run_experiment

Script Parameters
-----------------
    <name>: The name of the experiment to run (see README for more info)
    <config>...: Optional list of configuration file names relative to the
        directory of the experiment specified by <name>. Configuration options
        from these files will override the default options. Overriding happens
        in the order the arguments are given. Thus, if
            config1.yaml config2.yaml config3.yaml
        is the given argument, then config1.yaml options are applied, followed
        by config2.yaml, followed by config3.yaml. This means that arguments
        supplied later have higher priority when it comes to overriding.
    <group>: Optional name of a directory relative to the directory of the
        experiment specified by <name>. The experiment will be run once for each
        configuration file present in the <group> directory, with each run's
        configuration options being overridden by the corresponding configuration
        file.
"""
import argparse
import copy
import importlib
import os
from typing import Sequence, Mapping

import mlx


def _assign_key_value(keys, value, data, level):
    if len(keys) - level == 1:
        data[keys[level]] = value
    else:
        if keys[level] not in data:
            data[keys[level]] = {}
        _assign_key_value(keys, value, data[keys[level]], level + 1)


class ParseConfiguration(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not hasattr(namespace, 'extra_config'):
            namespace.extra_config = {}

        for value in values:
            if '=' not in value:
                raise argparse.ArgumentError(
                    self,
                    f"Invalid format; must be key=value."
                )

            key, val = value.split('=', 1)
            try:
                final_value = eval(val.strip())
            except NameError:
                final_value = val.strip()

            keys = key.split('.')
            _assign_key_value(keys, final_value, namespace.extra_config, 0)


def _run_cli():
    parser = argparse.ArgumentParser(
        prog='mlx.run',
        description='Run an experiment or a group of experiments.'
    )

    parser.add_argument(
        'name',
        help='The experiment name.'
    )

    parser.add_argument(
        'configs', nargs='*',
        help='Optional extra config files.'
    )

    parser.add_argument(
        '--group', '-g',
        help='Optional directory of config file, with each '
             'of which the experiment will be run.',
        default=None
    )
    
    parser.add_argument(
        '--in_group',
        help='Optional group for the experiment (unlike the --group '
             'argument, this just applies the given name but does not '
             'sequentially run a set of experiments in a group. To be '
             'used for running experiments in a group in parallel)',
        default=None
    )

    parser.add_argument(
        '--config',
        help='Optional configuration to override the loaded configuration. '
             'Use "." to set nested values, for example, a.b sets sub-property b '
             'of property a. Pass arguments using key=value format.',
        action=ParseConfiguration,
        nargs='*'
    )

    args = parser.parse_args()

    # Add cwd to path so that imports work correctly
    import sys
    sys.path.insert(0, os.getcwd())

    if hasattr(args, 'extra_config'):
        extra_config = args.extra_config
    else:
        extra_config = None

    run_experiment(args.name, args.configs, args.group, args.in_group, extra_config)


def run_experiment(
        name: str,
        configs: Sequence[str | os.PathLike] | None = None,
        group: str | os.PathLike | None = None,
        in_group: str | os.PathLike | None = None,
        extra_config: Mapping | None = None
):
    """
    Runs an experiment or a group of experiments.

    :param name: The experiment name (see README for more info)
    :param configs: Optional override configuration file names. Configuration
        options will override in the order the files are given, so files later
        in the list will override files earlier in the list.
    :param group: Optional directory containing configuration files. An
        experiment group with the same name will be created. The group will
        consist of runs of the same experiment, one for each configuration file
        in the directory, with the configuration options for that run being
        overridden by the corresponding configuration file.
    :param in_group: Optional name of group to apply to this experiment.
    :param extra_config: Optional dictionary of extra configuration values to
        apply after all other configuration values have been applied (including
        group member configuration).
    """
    # Handle no configs specified and convert to list so that we can prepend the
    # default config
    if configs is None:
        configs = ()
    configs = list(configs)

    # Determine experiment location and whether a main or side experiment was
    # specified
    experiment_dir = os.path.join('experiments', name)
    local_id = os.path.split(name)[-1]
    if os.path.isfile(experiment_dir + '.py'):
        # Side experiment was specified
        experiment_file = experiment_dir
        experiment_dir = os.path.dirname(experiment_dir)

        # Insert side experiment default config to be applied before any overrides
        configs.insert(0, local_id + '.config')
    else:
        # Main experiment was specified
        experiment_file = os.path.join(experiment_dir, local_id)

    # Insert default config to be applied first
    configs.insert(0, 'config')
    # Load experiment module and find experiment class
    experiment_module_name = '.'.join(os.path.normpath(experiment_file).split(os.sep))
    experiment_module = importlib.import_module(experiment_module_name)
    for var in vars(experiment_module).values():
        try:
            if issubclass(var, mlx.Experiment):
                experiment = var()
                break
        except TypeError:
            continue
    else:
        raise ValueError('No experiment class present in the given experiment module.')

    # Get base config
    base_config = mlx.config.load_base_config()
    for config_file in configs:
        if config_file.startswith('global:'):
            global_file = os.path.join('experiments', config_file[len('global:'):])
            config = mlx.config.load_config(global_file)
        else:
            try:
                config = mlx.config.load_config(
                    os.path.join(experiment_dir, config_file),
                    raise_exc=True
                )
            except mlx.config.ConfigNotFoundError:
                config = mlx.config.load_config(os.path.join('experiments', config_file))
        mlx.config.config_update_recursive(base_config, config, default_option='add')

    # Run experiment/experiment group
    if group is None:
        if extra_config is not None:
            mlx.config.config_update_recursive(base_config, extra_config, default_option='add')
        experiment.run(base_config, name=name, group=in_group)
    else:
        # Load group configurations
        group_configs = []
        for file in os.listdir(os.path.join(experiment_dir, group)):
            try:
                config = mlx.config.load_config(
                    os.path.join(experiment_dir, group, file),
                    raise_exc=True
                )
                config['group_id'] = os.path.splitext(file)[0]
                group_configs.append(config)
            except mlx.config.ConfigNotFoundError:
                print('Failed')

        # Run experiments
        experiment.start_group()
        for config in group_configs:
            # Load run config and override base config. Make sure to deeply copy
            # base config so that options from one run don't bleed into
            # subsequent ones
            run_config = copy.deepcopy(base_config)
            mlx.config.config_update_recursive(run_config, config, default_option='add')
            if extra_config is not None:
                mlx.config.config_update_recursive(run_config, extra_config, default_option='add')
            experiment.run(run_config, name=name, group=group)
        experiment.finish_group()


if __name__ == '__main__':
    _run_cli()
