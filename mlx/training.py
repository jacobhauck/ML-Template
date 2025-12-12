import os
import signal
import time
from abc import ABC, abstractmethod
from enum import Enum

import torch

import mlx


class Verbosity(Enum):
    SILENT = 0
    NORMAL = 1
    VERBOSE = 2
    VERY_VERBOSE = 3


class DelayedKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = None
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        # noinspection PyAttributeOutsideInit
        self.signal_received = (sig, frame)

    def __exit__(self, _type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received is not None:
            self.old_handler(*self.signal_received)


class BaseTrainer(ABC):
    def __init__(self, config, run, save_interval=None, verbosity=1):
        self.run = run
        self.config = config
        self.data_seen = 0
        self.device = config.get('device', 'cpu')
        self.save_interval = save_interval
        self.verbosity = verbosity

        if self.verbosity >= Verbosity.NORMAL.value:
            run_type = 'local' if isinstance(self.run, mlx.LocalRun) else 'W&B'
            print(f'Initializing {run_type} run {self.run.id}')

        self.model = mlx.create_module(config['model']).to(self.device)
        if self.verbosity >= Verbosity.VERBOSE.value:
            print('Model initialized')

        self.optim = mlx.create_optimizer(self.model.parameters(), config['optim'])
        if self.verbosity >= Verbosity.VERBOSE.value:
            print('Optimizer initialized')

        if 'lr_scheduler' in config:
            self.lr_scheduler = mlx.create_lr_scheduler(self.optim, config['lr_scheduler'])
            if self.verbosity >= Verbosity.VERBOSE.value:
                print('Learning rate scheduler initialized')
        else:
            self.lr_scheduler = None

        self.datasets, self.data_loaders = self.load_datasets(config)
        if self.verbosity >= Verbosity.VERBOSE.value:
            print(f'{len(self.datasets)} datasets loaded')

        if self.run.step is not None and self.run.step > 0:
            self.load_checkpoint(self.run.step)

    def checkpoint_path(self, step=None):
        if step is None:
            step = self.run.step

        assert step is not None and step > 0, \
            'Cannot save until at least one step is made'

        return os.path.join(mlx.CHECKPOINT_DIR, f'{self.run.id}-{step}.pth')

    def save_checkpoint(self):
        state = {
            'step': self.run.step,
            'data_seen': self.data_seen,
            'model': self.model.cpu().state_dict(),
            'optim': mlx.optimizer_to(self.optim, 'cpu').state_dict(),
        }

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = mlx.optimizer_to(self.lr_scheduler, 'cpu').state_dict()

        torch.save(state, self.checkpoint_path())

        if self.verbosity >= Verbosity.NORMAL.value:
            print(f'Saved checkpoint on step {self.run.step}')

    def load_checkpoint(self, step):
        state = torch.load(self.checkpoint_path(step))

        self.data_seen = state['data_seen']

        self.model.cpu().load_state_dict(state['model'])
        self.model.to(self.device)
        torch.cuda.empty_cache()

        mlx.optimizer_to(self.optim, 'cpu').load_state_dict(state['optim'])
        mlx.optimizer_to(self.optim, self.device)

        if self.lr_scheduler is not None:
            mlx.lr_scheduler_to(self.lr_scheduler, 'cpu').load_state_dict(state['lr_scheduler'])
            mlx.lr_scheduler_to(self.lr_scheduler, self.device)

        if self.verbosity >= Verbosity.NORMAL.value:
            print(f'Loaded checkpoint {step}')

    def get_interruption_selection(self):
        print(f'Training terminated early on step {self.run.step}')
        print('Mark as "Killed" or "Finished"? (default = "Finished"; killed runs *cannot* be resumed)')
        while True:
            selection = input().lower()
            if selection in ('', 'killed', 'finished'):
                break

        return selection

    def train_one_step(self, epoch, batch, data):
        self.optim.zero_grad()
        prediction, losses = self.loss(data)
        losses['objective'].backward()
        self.optim.step()

        log = {'epoch': epoch, 'batch': batch, 'data_seen': self.data_seen}
        log.update({name: loss.item() for name, loss in losses.items()})
        metrics = self.metrics(prediction, data)
        log.update(metrics)
        log.update(self.lr_scheduler_log())
        self.run.log(log)

        self.data_seen += self.get_batch_size(data)

        if self.verbosity >= Verbosity.VERBOSE.value:
            print(f'Epoch {epoch}, batch {batch}, data seen {self.data_seen}: '
                  f'{losses["objective"].item():.06f}')

        if self.verbosity >= Verbosity.VERY_VERBOSE.value:
            print('Losses')
            for key, value in losses.items():
                print(f'    {key}: {value.item():.06f}')
            print('Metrics')
            for key, value in metrics.items():
                print(f'    {key}: {value:.06f}')

    def num_steps(self, epochs=1):
        return epochs * len(self.data_loaders['train'])

    def train(self, epochs=1):
        marker = time.time()
        if self.run.step is not None and self.run.step > 0:
            epochs_left = (self.num_steps(epochs) - self.run.step) // self.num_steps()
            if self.verbosity >= Verbosity.NORMAL.value:
                print(f'Resuming training from step {self.run.step}')
                print(f'Approximately {epochs_left} epochs remaining to reach target {epochs}')
                print(f'Training steps: {self.num_steps(epochs_left)}')
        else:
            epochs_left = epochs
            if self.verbosity >= Verbosity.NORMAL.value:
                print(f'Starting training from step 0 for {epochs} epochs')
                print(f'Training steps: {self.num_steps(epochs_left)}')

        try:
            for epoch in range(epochs_left):
                if self.verbosity >= Verbosity.VERBOSE.value:
                    print(f'Starting epoch {epoch}')

                if self.verbosity >= Verbosity.VERY_VERBOSE.value:
                    print(f'Total data seen {self.data_seen}')

                for batch, data in enumerate(self.data_loaders['train']):
                    # Delay keyboard interrupts until the end of a training step
                    with DelayedKeyboardInterrupt():
                        self.train_one_step(epoch, batch, data)

                        if self.save_interval is not None and \
                                time.time() - marker > self.save_interval:
                            marker = time.time()
                            self.save_checkpoint()

                        if batch == len(self.data_loaders['train']) - 1 and \
                                self.lr_scheduler is not None:
                            self.lr_scheduler.step()

        except KeyboardInterrupt as e:
            selection = self.get_interruption_selection()

            if selection == 'killed':
                raise e

        if self.verbosity >= Verbosity.NORMAL.value:
            print(f'Saving final checkpoint')
        self.save_checkpoint()

    # noinspection PyMethodMayBeStatic
    def get_batch_size(self, data):
        return len(data[0])

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def metrics(self, prediction, data):
        return {}

    # noinspection PyMethodMayBeStatic
    def lr_scheduler_log(self):
        return {}

    @abstractmethod
    def loss(self, data):
        """
        Compute the loss for given data
        :param data: One batch; whatever is produced by DataLoader
        :return: tuple (prediction, losses). prediction is the model output
            (any type). losses is a dictionary of losses that must contain
            key "objective", which is the loss that is used for training
        """
        raise NotImplemented

    @abstractmethod
    def load_datasets(self, config):
        """
        Load datasets and data loaders for the trainer
        :param config: The run config
        :return: tuple (datasets, data_loaders), each of which is a dictionary
            whose keys are strings and values are Dataset/DataLoader objects.
            DataLoader with key 'train' will be used for training
        """
        raise NotImplemented
