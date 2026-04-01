from typing import Mapping

"""
Dictionary of custom, named LR schedulers. If an LR scheduler is requested in
create_lr_scheduler that is not present in the torch.optim.lr_scheduler namespace,
then create_lr_scheduler will look for the module in this dictionary.
"""
custom_lr_schedulers = {  # dict[str, LRScheduler]

}


def create_lr_scheduler(optimizer, config: Mapping):
    """
    Create an LR scheduler from its configuration.

    :param optimizer: Optimizer to assign the scheduler to
    :param config: Dictionary of LR scheduler configuration options. Must
        contain a 'name' field to specify which scheduler to load, which refers
        to the name of the optimizer within the torch.optim.lr_scheduler module,
        or else the name used as a key in the lr_scheduler.custom_lr_schedulers
        dictionary of custom schedulers.
    :return: An instance of the LR scheduler specified by the given
        configuration settings.
    """
    import torch

    config = dict(config)
    name = config.pop('name')

    module = getattr(torch.optim.lr_scheduler, name, None)
    if module is None:
        module = custom_lr_schedulers[name]

    return module(optimizer, **config)


# noinspection PyProtectedMember
def lr_scheduler_to(lr_scheduler, device):
    if hasattr(lr_scheduler, 'state'):
        for param in lr_scheduler.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for sub_param in param.values():
                    if isinstance(sub_param, torch.Tensor):
                        sub_param.data = sub_param.data.to(device)
                        if sub_param._grad is not None:
                            sub_param._grad.data = sub_param._grad.data.to(device)

    return lr_scheduler
