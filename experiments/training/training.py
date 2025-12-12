import torch.utils.data
import mlx


class DemoDataset(torch.utils.data.Dataset):
    def __init__(self, size, d_in):
        self.x = torch.randn((size, d_in))
        self.y = torch.norm(self.x, dim=1, keepdim=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class DemoTrainer(mlx.training.BaseTrainer):
    def load_datasets(self, config):
        dataset = DemoDataset(config['dataset']['size'], config['dataset']['d_in'])
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['trainer']['batch_size'],
            shuffle=config['trainer'].get('shuffle', True)
        )
        return {'train': dataset}, {'train': data_loader}

    def loss(self, data):
        x, y = data
        y_pred = self.model(x)
        return y_pred, {'objective': torch.nn.functional.mse_loss(y_pred, y)}

    def metrics(self, prediction, data):
        return {'MAE': torch.nn.functional.l1_loss(prediction, data[1])}


class TrainingExperiment(mlx.WandBExperiment):
    def wandb_run(self, config, run) -> None:
        trainer = DemoTrainer(
            config, run,
            save_interval=config['trainer']['save_interval'],
            verbosity=config['trainer'].get('verbosity', 1)
        )
        trainer.train(config['trainer']['epochs'])
