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
    def __init__(
            self,
            config,
            run,
            save_interval=None,
            log_interval=None,
            verbosity=1
    ):
        self.run = run
        self.config = config
        self.data_seen = 0
        self.device = config.get('device', 'cpu')
        self.save_interval = save_interval
        self.log_interval = log_interval
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
            self.load_checkpoint(self.run.step - 1)

    def checkpoint_path(self, step=None, must_exist=False):
        if step is None:
            step = self.run.step

        assert step is not None and step > 0, \
            'Cannot save until at least one step is made'
        
        check_path = os.path.join(mlx.CHECKPOINT_DIR, f'{self.run.id}-{step}.pth')
        if must_exist and not os.path.exists(check_path):
            print('Failed to find requested checkpoint. Falling back to most recent')
            
            latest_step = -1
            latest_file = None
            for file in os.listdir(mlx.CHECKPOINT_DIR):
                if file.startswith(f'{self.run.id}'):
                    file_step = int(os.path.splitext(file)[0].split('-')[-1])
                    if file_step > latest_step:
                        latest_step = file_step
                        latest_file = file
            
            if latest_file is None:
                raise IOError('Could not find any checkpoints!')
            
            return os.path.join(mlx.CHECKPOINT_DIR, latest_file)
        else:
            return check_path

    def save_checkpoint(self):
        state = {
            'step': self.run.step,
            'data_seen': self.data_seen,
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict()
        }
        
        state.update(self.dump_additional_state())

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = mlx.optimizer_to(self.lr_scheduler, 'cpu').state_dict()
        
        path = self.checkpoint_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)

        if self.verbosity >= Verbosity.NORMAL.value:
            print(f'Saved checkpoint on step {self.run.step}')

    def load_checkpoint(self, step):
        state = torch.load(self.checkpoint_path(step, True))

        self.data_seen = state['data_seen']

        self.model.load_state_dict(state['model'])

        if self.lr_scheduler is not None:
            mlx.lr_scheduler_to(self.lr_scheduler, self.device)
        
        self.load_additional_state(state)
        
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

    def train_one_step(self, epoch, batch, data, do_log=True, log_step=None):
        self.optim.zero_grad()
        prediction, losses = self.loss(data)
        losses['objective'].backward()
        self.optim.step()

        metrics = self.metrics(prediction, data)

        if do_log:
            log = {'epoch': epoch, 'batch': batch, 'data_seen': self.data_seen}
            log.update({name: loss.item() for name, loss in losses.items()})
            log.update(metrics)
            log.update(self.lr_scheduler_log())
            self.run.log(log, step=log_step)

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
        self.model.train(True)
        save_marker = time.time()
        log_marker = time.time()
        log_next = False
        log_step = 0 if self.run.step is None else self.run.step

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
                        self.train_one_step(epoch, batch, data, do_log=log_next, log_step=log_step)
                        log_step += 1
                        log_next = False

                        if self.save_interval is not None and \
                                time.time() - save_marker > self.save_interval:
                            self.save_checkpoint()
                            save_marker = time.time()

                        if self.log_interval is not None and \
                                time.time() - log_marker > self.log_interval:
                            log_next = True
                            log_marker = time.time()

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
    
    def evaluate(self, datasets=('train',)):
        self.model.train(False)
        
        losses_by_dataset = {}
        metrics_by_dataset = {}
        for name in datasets:
            all_losses = {}
            all_metrics = {}
            data_loader = torch.utils.data.DataLoader(self.datasets[name], batch_size=1)
            for data in data_loader:
                with torch.no_grad():
                    prediction, losses = self.loss(data)
                    metrics = self.metrics(prediction, data)
                
                if len(all_losses) == 0:
                    all_losses = {key: [] for key in losses}
                    all_metrics = {key: [] for key in metrics}
                
                for key, value in losses.items():
                    all_losses[key].append(value.item())
                
                for key, value in metrics.items():
                    all_metrics[key].append(value.item())
            
            losses_by_dataset[name] = {key: torch.tensor(value) for key, value in all_losses.items()}
            metrics_by_dataset[name] = {key: torch.tensor(value) for key, value in all_metrics.items()}
        
        return losses_by_dataset, metrics_by_dataset
    
    # noinspection PyMethodMayBeStatic
    def get_batch_size(self, data):
        return len(data[0])

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def metrics(self, prediction, data):
        return {}

    # noinspection PyMethodMayBeStatic
    def lr_scheduler_log(self):
        return {}
    
    # noinspection PyMethodMayBeStatic
    def dump_additional_state(self):
        return {}
    
    # noinspection PyMethodMayBeStatic
    def load_additional_state(self, state):
        pass
    
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
