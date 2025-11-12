"""
Base utilities for managing and running experiments.

Classes
-------
    - Experiment: base class for all experiments

    - WandBExperiment: base class for all experiments that are logged to W&B
"""
import abc
import os
import uuid
from typing import Mapping

import wandb
import yaml


CHECKPOINT_DIR = '.checkpoints'
LOCAL_RUNS_DIR = os.path.join(CHECKPOINT_DIR, 'local')


if os.path.exists('wandb_config.yaml'):
    with open('wandb_config.yaml') as f:
        wandb_config = yaml.safe_load(f)
        assert 'entity' in wandb_config, 'invalid W&B config'
        assert 'project' in wandb_config, 'invalid W&B config'
else:
    wandb_config = None


def wandb_path():
    global wandb_config
    assert wandb_config is not None, 'No W&B config found!'
    return wandb_config['entity'] + '/' + wandb_config['project']


class Experiment(abc.ABC):
    """
    Represents a type of experiment.

    Used for automatic running in run_experiment. All experiment classes should
    be subclasses of Experiment.
    """
    started_group = False

    @abc.abstractmethod
    def run(self, config: Mapping, name: str, group: str | None = None) -> None:
        raise NotImplemented

    def start_group(self):
        self.started_group = True

    def finish_group(self):
        self.started_group = False


class _DummyRunAttribute:
    def __call__(self, *args, **kwargs):
        pass


class LocalRun:
    _attr = _DummyRunAttribute()

    def __init__(self, resume_id=None):
        if resume_id is None:
            self._id = str(uuid.uuid4())
            self.resumed = False
            self.step = 0
            os.makedirs(LOCAL_RUNS_DIR, exist_ok=True)
        else:
            self._id = resume_id
            self.resumed = True

            with open(self._step_file):
                self.step = int(f.read())

    @property
    def _step_file(self):
        return os.path.join(LOCAL_RUNS_DIR, self._id)

    def log(self, *_, **__):
        self.step += 1
        with open(self._step_file, 'w') as step_file:
            step_file.write(str(self.step))

    def __getattr__(self, item):
        """This is a cheap way to provide some compatibility with W&B interface"""
        if item == 'id':
            return self._id

        return self._attr


class WandBExperiment(Experiment):
    is_first_group_experiment = False

    def start_group(self):
        self.is_first_group_experiment = True

    def run(self, config: Mapping, name: str, group: str | None = None) -> None:
        assert wandb_config is not None, \
            'Must add properly formed wandb_config.yaml file to run W&B experiments'

        if 'use_wandb' in config and not config['use_wandb']:
            run = LocalRun(resume_id=config.get('resume_id'))
        elif 'resume_id' in config:
            run_id = config['resume_id']
            run = wandb.init(
                entity=wandb_config['entity'],
                project=wandb_config['project'],
                id=run_id,
                resume='must'
            )
            config = run.config
        else:
            api = wandb.Api()
            runs = api.runs(
                wandb_path(),
                filters=dict(
                    displayName={'$regex': rf'{name}.*'}
                )
            )
            run_number = 1 + max((int(run.name.split('-')[1]) for run in runs), default=0)
            group_number = 1 + max(
                (int(run.group.split('-')[2]) for run in runs if run.group != name),
                default=0
            )
            if self.is_first_group_experiment:
                self.is_first_group_experiment = False
            else:
                group_number -= 1

            run = wandb.init(
                entity=wandb_config['entity'],
                project=wandb_config['project'],
                config=dict(config),
                name=f'{name}-{run_number}',
                group=f'{name}-{group}-{group_number}' if group is not None else name
            )

        self.wandb_run(config, run)

        # noinspection PyArgumentList
        run.finish()

    @abc.abstractmethod
    def wandb_run(self, config: Mapping, run) -> None:
        raise NotImplemented


class TrainingExperiment(WandBExperiment):
    def wandb_run(self, config: Mapping, run) -> None:
        training_config = dict(config['trainer'])
        training_config['config'] = config
        training_config['run'] = run
        trainer = mlx.create_module(training_config)

        trainer.train(config['training_epochs'])
