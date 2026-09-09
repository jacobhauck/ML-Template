import random
import os


def subset_indices(config, dataset):
    sample_cfg = config.get('sample', {})
    if 'indices' in sample_cfg:
        return list(sample_cfg['indices'])
    elif 'start_index' in sample_cfg:
        stop_index = sample_cfg.get('stop_index', len(dataset) - 1)
        step = sample_cfg.get('index_step', 1)
        return list(range(sample_cfg['start_index'], stop_index + 1, step))
    elif sample_cfg.get('random', False):
        size = sample_cfg.get('size', len(dataset))
        return random.sample(range(len(dataset)), k=size)
    else:
        size = sample_cfg.get('size', len(dataset))
        return list(range(size))


def results_dir(name, sub_path=None):
    output_dir = os.path.join('results', name)
    if sub_path is not None:
        output_dir = os.path.join(output_dir, sub_path)

    os.makedirs(output_dir, exist_ok=True)

    return output_dir
