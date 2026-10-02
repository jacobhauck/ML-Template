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

def configure_plotting(config):
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
    import matplotlib.pyplot as plt

    if config.get('show', False):
        plt.show()

    ext = config.get('format', 'png')
    output_file = os.path.join(results_dir(name, sub_path), file_name + '.' + ext)
    fig.savefig(output_file, bbox_inches='tight')
    print(f'Saved figure: {output_file}')
