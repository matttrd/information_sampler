import argparse
import os
import pickle as pkl
import numpy as np

from IPython import embed
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def create_histograms(counts):
    print('\nNote, the epochs are not parametrized in the function: create_histograms\n')
    x_grid = np.linspace(0, 250, 250) # you can choose the amplitude chaning 250
    for exp, value in counts.items():
        plt.figure()
        plt.title(f'KDE examples count')
        for c in value['counts']:
            pdf = kde_sklearn(c, x_grid, bandwidth=2)  #Note, the bandwidth can be changed here
            plt.plot(x_grid, pdf, label="Mean: " + str(np.mean(np.array(c))) + " std: " + str(np.std(np.array(c))) + f', epoch: {str(sum(c)/50000)}', linewidth=2)
            plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(value['save_path'], 'kde.pdf'), dpi=2560, bbox_inches='tight')

            # plt.figure()
            # plt.hist(c, bins=200)
            # plt.savefig(os.path.join(value['save_path'], 'histogram.pdf', dpi=2560, bbox_inches='tight'))




parser = argparse.ArgumentParser(description='count_visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--sd', default=f'..{os.sep}results') #Note this is not used now since we are saving on the exp run folder 
parser.add_argument('--base', default=f'..{os.sep}results')
parser.add_argument('--epochs', default=[0,1,2,59,119,159])
opt = vars(parser.parse_args())

epochs = [str(ep) for ep in opt['epochs']]

# Parsing the folder with experiments creating a dict,
# key=experiment, value=list of differnt runs (!=hyperparameters)
exp_type = {}
for exp in os.listdir(opt['base']):
    experiments = []
    for filename in os.listdir(os.path.join(opt['base'], exp)):
        experiments.append(filename)
    exp_type[exp] = experiments

# Creating a dictionary (keys are exp + run) of dictionaries containing the
# actual counts (filtered by epochs) and the saving path
counts = {}
for exp, run in exp_type.items():
    for name_run in run:
        path = os.path.join(opt['base'], exp, name_run)
        counts[exp + '_' + name_run] = {'counts': []}
        for file in os.listdir(os.path.join(path, 'sample_counts')):
            if file.split('_')[-1].split('.')[0] in epochs:
                with open(os.path.join(path, 'sample_counts', file), 'rb') as f:
                    counts[exp + '_' + name_run]['counts'].append(pkl.load(f))
        counts[exp + '_' + name_run]['save_path'] = os.path.join(path, 'analysis')
        if not os.path.isdir(counts[exp + '_' + name_run]['save_path']):
            os.mkdir(counts[exp + '_' + name_run]['save_path'])

create_histograms(counts)
