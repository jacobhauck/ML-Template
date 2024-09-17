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
from typing import Sequence
import os
import utils.config
import importlib
import experiments
import copy


def run_experiment(
        name: str,
        configs: Sequence[str | os.PathLike] | None = None,
        group: str | os.PathLike | None = None
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
    """
    # Handle no configs specified and convert to list so that we can prepend the
    # default config
    if configs is None:
        configs = ()
    configs = list(configs)

    # Determine experiment location and whether a main or side experiment was
    # specified
    experiment_dir = os.path.join('experiments', name)
    if os.path.isfile(experiment_dir):
        # Side experiment was specified
        experiment_file = experiment_dir
        experiment_dir = os.path.dirname(experiment_dir)

        # Insert side experiment default config to be applied before any overrides
        configs.insert(0, os.path.join(experiment_dir, name + '.config'))
    else:
        # Main experiment was specified
        experiment_file = os.path.join(experiment_dir, name)

    # Insert default config to be applied first
    configs.insert(0, os.path.join(experiment_dir, 'config'))

    # Load experiment module and find experiment class
    experiment_module_name = '.'.join(os.path.split(experiment_file))
    experiment_module = importlib.import_module(experiment_module_name)
    for var in vars(experiment_module).values():
        if isinstance(var, experiments.Experiment):
            experiment = var
            break
    else:
        raise ValueError('No experiment class present in the given experiment module.')

    # Get base config
    base_config = utils.config.load_base_config()
    utils.config.load_config(os.path.join(experiment_dir, ''))
    for config_file in configs:
        config = utils.config.load_config(os.path.join(experiment_dir, config_file))
        utils.config.config_update_recursive(base_config, config, default_option='add')

    # Run experiment/experiment group
    if group is None:
        experiment.run(base_config)
    else:
        for config_file in group:
            # Load run config and override base config. Make sure to deeply copy
            # base config so that options from one run don't bleed into
            # subsequent ones
            run_config = copy.deepcopy(base_config)
            config = utils.config.load_config(os.path.join(experiment_dir, config_file))
            utils.config.config_update_recursive(run_config, config, default_option='add')

            experiment.run(run_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_experiment',
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

    args = parser.parse_args()
    run_experiment(args.name, args.configs, args.group)
