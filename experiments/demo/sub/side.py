"""Simple demo experiment for testing run_experiment script."""
from mlx import WandBExperiment

from ..demo import _print


class DemoSide(WandBExperiment):
    """Demonstration experiment that simply prints the provided config."""

    def wandb_run(self, config, run):
        print('Running demo SUB-SIDE Experiment.')
        print(run.name)
        _print(config)
