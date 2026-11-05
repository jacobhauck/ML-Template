import random
import os
import torch


def subset_indices(config, dataset):
    """
    Gets indices of a sample in a given dataset based on given
    sample configuration
    
    :param config: Sample configuration dictionary. Should contain a
                   top-level entry with key 'sample'; if not, then the
                   returned sample indices are the whole dataset. If
                   config['sample'] is given, it must be a dictionary with
                   one of the four following sets of keys, parsed with
                   priority given in the same order. First, the key 
                   'indices' may be used with a value giving an explicit
                   array of indices to return. Second, the keys 'start_index'
                   and 'stop_index' may be used to specify a contiguous 
                   range of indices, which will be capped at the size of 
                   the dataset and is inclusive on both sides; the range can
                   be made non-contiguous, taking data in strides of a
                   size given by the key 'index_step', which is optional.
                   Third, the key 'random' can be given with the value True,
                   in which case all indices will be returned in a random 
                   order; if the key 'size' is present, then its value will
                   specify the size of the random sample. Fourth, the key 
                   'size' can be given on its own (or with 'random' having
                   the value False), in which case only the given number of
                   indices in a contiguous range starting from 0 will be
                   returned.
    :param dataset: The dataset from which to take indices
    :return: List of indices into the given dataset chosen according to the
             given configuration
    """
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
    """
    Gets the path to the results directory for a given experiment with
    optional sub-path. Creates this directory (and any parent directories)
    if it does not exist.
    
    :param name: experiment name
    :param sub_path: sub-path of the desired results directory relative to
                     the experiment results directory
    :return: path of the desired experiment results directory
    """
    output_dir = os.path.join('results', name)
    if sub_path is not None:
        output_dir = os.path.join(output_dir, sub_path)

    os.makedirs(output_dir, exist_ok=True)

    return output_dir

def configure_plotting(config):
    """
    Applies global style configuration options to matplotlib
    based on the given configuration
    """
    import matplotlib.pyplot as plt

    if 'fonts' in config:
        fonts = config['fonts']
        if 'size' in fonts:
            plt.rc('font', size=fonts['size'])  # controls default text sizes
        if 'axis_title_size' in fonts:
            plt.rc('axes', titlesize=fonts['axis_title_size'])  # font size of the axes title
        if 'axis_label_size' in fonts:
            plt.rc('axes', labelsize=fonts['axis_label_size'])  # font size of the x and y labels
        if 'xtick_size' in fonts:
            plt.rc('xtick', labelsize=fonts['xtick_size'])  # font size of the tick labels
        if 'ytick_size' in fonts:
            plt.rc('ytick', labelsize=fonts['ytick_size'])  # font size of the tick labels
        if 'legend_size' in fonts:
            plt.rc('legend', fontsize=fonts['legend_size'])  # legend font size
        if 'fig_title_size' in fonts:
            plt.rc('figure', titlesize=fonts['fig_title_size'])  # font size of the figure title
        if 'family' in fonts:
            plt.rc('font', family=fonts['family'])
        if 'use_tex' in fonts:
            plt.rc('text', usetex=fonts['use_tex'])


def show_and_save(fig, file_name, config, name, sub_path=None):
    """
    Shows a figure (optionally) and saves it
    :param fig: The figure to show
    :param file_name: The name of the output file
    :param config: experiment configuration, which specifies various options like
                   output file format and whether to show the figure, and output location
    :param name: experiment name
    :param sub_path: optional sub-path in the experiment results directory in which
                     to save the file
    """
    import matplotlib.pyplot as plt

    if config.get('show', False):
        plt.show()

    ext = config.get('format', 'png')
    output_file = os.path.join(results_dir(name, sub_path), file_name + '.' + ext)
    fig.savefig(output_file, bbox_inches='tight')
    print(f'Saved figure: {output_file}')


def print_performance_summary(eval_losses, eval_metrics):
    """
    Prints a complete summary of the performance of a model
    :param eval_losses: Dictionary of loss values returned by BaseTrainer.evaluate
    :param eval_metrics: Dictionary of metric values returned by BaseTrainer.evaluate
    """
    print('======= Losses =======')
    for dataset, losses in eval_losses.items():
        print(f'    ===== Dataset: {dataset} =====')
        for loss_name, loss in losses.items():
            print(f'        === Loss: {loss_name} ===')
            print(f'        Q1:     {float(torch.quantile(loss, 0.25))}')
            print(f'        Median: {float(torch.quantile(loss, 0.5))}')
            print(f'        Q3:     {float(torch.quantile(loss, 0.75))}')
            print(f'        Avg:    {float(torch.mean(loss))}')
            print(f'        Std:    {float(torch.std(loss))}')
            print()
        print()
    print()

    print('======= Metrics =======')
    for dataset, metrics in eval_metrics.items():
        print(f'    ===== Dataset: {dataset} =====')
        for metric_name, metric in metrics.items():
            print(f'        === Loss: {metric_name} ===')
            print(f'        Q1:     {float(torch.quantile(metric, 0.25))}')
            print(f'        Median: {float(torch.quantile(metric, 0.5))}')
            print(f'        Q3:     {float(torch.quantile(metric, 0.75))}')
            print(f'        Avg:    {float(torch.mean(metric))}')
            print(f'        Std:    {float(torch.std(metric))}')
            print()
        print()
    print()
