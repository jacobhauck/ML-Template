"""Simple demo experiment for testing run_experiment script."""
from experiments import WandBExperiment

from .demo import _print


class DemoSide(WandBExperiment):
    """Demonstration experiment that simply prints the provided config."""

    def wandb_run(self, config, run):
        print('Running demo SIDE Experiment.')
        print(run.name)
        _print(config)
