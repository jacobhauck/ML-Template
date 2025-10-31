from abc import ABC, abstractmethod
import signal
import torch
import mlx
import time


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
    def __init__(self, config, run, save_interval=None):
        self.run = run
        self.config = config
        self.data_seen = 0
        self.device = config.get('device', 'cpu')
        self.save_interval = save_interval
        self.model = mlx.create_module(config['model']).to(self.device)
        self.optim = mlx.create_optimizer(self.model.parameters(), config['optim'])

        if 'lr_scheduler' in config:
            self.lr_scheduler = mlx.create_lr_scheduler(self.optim, config['lr_scheduler'])
        else:
            self.lr_scheduler = None

        self.datasets, self.data_loaders = self.load_datasets(config)

    def checkpoint_path(self, step=None):
        assert step > 0, 'Cannot save until at least one step is made'

        if step is None:
            step = self.step

        return os.path.join(mlx.CHECKPOINT_DIR, f'{self.run.id}-{step}.pth')

    def save_checkpoint(self):
        state = {
            'step': self.step,
            'data_seen': self.data_seen,
            'model': self.model.cpu().state_dict(),
            'optim': mlx.optimizer_to(self.optim, 'cpu').state_dict(),
        }

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = mlx.optimizer_to(self.lr_scheduler, 'cpu').state_dict()

        torch.save(state, self.checkpoint_path())

    def load_checkpoint(self, step):
        state = torch.load(self.checkpoint_path(step))

        self.data_seen = state['data_seen']

        self.model.cpu().load_state_dict(state['model']).to(self.device)
        torch.cuda.empty_cache()

        mlx.optimizer_to(self.optim, 'cpu').load_state_dict(state['optim'])
        mlx.optimizer_to(self.optim, self.device)

        if self.lr_scheduler is not None:
            mlx.lr_scheduler_to(self.lr_scheduler, 'cpu').load_state_dict(state['lr_scheduler'])
            mlx.lr_scheduler_to(self.lr_scheduler, self.device)

    @staticmethod
    def get_interruption_selection():
        print(f'Training terminated early on step {run.summary["_step"]}')
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
        log.update(self.model.metrics(prediction, data))
        log.update(self.lr_scheduler_log())
        self.run.log(log)

        self.data_seen += self.get_batch_size(data)

    def train(self, epochs=1):
        marker = time.time()
        try:
            for epoch in range(epochs):
                for batch, data in enumerate(self.data_loaders['train']):
                    # Delay keyboard interrupts until the end of a training step
                    with DelayedKeyboardInterrupt():
                        self.train_one_step(epoch, batch, data)

                        if self.save_interval is not None and \
                                time.time() - marker > self.save_interval:
                            marker = time.time()

                        if batch == len(self.data_loaders['train']) - 1 and \
                                self.lr_scheduler is not None:
                            self.lr_scheduler.step()

        except KeyboardInterrupt as e:
            selection = self.get_interruption_selection()

            if selection == 'killed':
                raise e

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
        raise NotImplemented

    @abstractmethod
    def load_datasets(self, config):
        raise NotImplemented
