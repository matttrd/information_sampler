import argparse
import os
import pickle as pkl
import numpy as np
import json

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
    for exp, value in counts.items():
        max_num_counts = int(np.array(counts[exp]['counts']).max())
        if max_num_counts < 500:
            max_num_counts = 500
        x_grid = np.linspace(0, max_num_counts, max_num_counts) # you can choose the amplitude changing 500
        plt.figure()
        info_exp = counts[exp]['name_exp']
        plt.title(info_exp['dataset'] + ' ' + info_exp['arch'] + ' BS' + str(info_exp['b']) + ' ' + info_exp['sampler'] + ' norm ' + str(info_exp['normalizer']) + ' Temp' + str(info_exp['temperature']) )
        for c in value['counts']:
            pdf = kde_sklearn(c, x_grid, bandwidth=2)  #Note, the bandwidth can be changed here
            plt.plot(x_grid, pdf, label="Mean: " + str(np.mean(np.array(c))) + " std: " + str(np.std(np.array(c))) + f', epoch: {str(sum(c)/50000)}', linewidth=2)
            plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(value['save_path'], 'kde.pdf'), dpi=2560, bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.title(info_exp['dataset'] + ' ' + info_exp['arch'] + ' BS' + str(info_exp['b']) + ' ' + info_exp['sampler'] + ' norm ' + str(info_exp['normalizer']) + ' Temp' + str(info_exp['temperature']) )
        for c in value['counts']:
            plt.hist(c, bins=x_grid, alpha=0.5, label="Mean: " + str(np.mean(np.array(c))) + " std: " + str(np.std(np.array(c))) + f', epoch: {str(sum(c)/50000)}', linewidth=2)
            plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(value['save_path'], 'histogram.pdf'), dpi=2560, bbox_inches='tight')
        plt.close()


            # plt.figure()
            # plt.hist(c, bins=200)
            # plt.savefig(os.path.join(value['save_path'], 'histogram.pdf', dpi=2560, bbox_inches='tight'))

def get_params_from_log(f):
    blacklist = ['lrs', 'B', 'd', 's', 'ids', 'd', \
                'save', 'metric', 'nc', 'g', 'j', 'env', 'burnin', \
                'alpha', 'delta', 'beta', 'lambdas', 'append', \
                'ratio', 'bfrac', 'l2', 'v', 'frac', 'rhos']
    r = {}
    for l in open(f,'r', encoding='latin1'):
        if '[OPT]' in l[:5]:
            r = json.loads(l[5:-1])
            fn = r['filename']
            for k in blacklist:
                r.pop(k, None)
            #r = {k: v for k,v in r.items() if k in whitelist}
            r['t'] = fn[fn.find('(')+1:fn.find(')')]
            return r
    assert len(r.keys) > 0, 'Could not find [OPT] marker in '+f

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='count_visualization',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sd', default=f'..{os.sep}results') #Note this is not used now since we are saving on the exp run folder
    parser.add_argument('--exp', default=f'..{os.sep}results', required=True)
    parser.add_argument('--epochs', default=[59,119,159,179])
    opt = vars(parser.parse_args())

    epochs = [str(ep) for ep in opt['epochs']]

    exp_path = os.path.join(f'..{os.sep}results{os.sep}', opt['exp'])

    # Parsing the experiment folder creating a dict of runs
    exp_runs = {}
    for exp in os.listdir(exp_path):
        if not exp == 'analysis_experiments':
            exp_runs[exp] = exp

    # embed()

    # Creating a dictionary (keys are exp_runs) of dictionaries containing the
    # actual counts (filtered by epochs) and the saving path
    counts = {}
    for exp, run in exp_runs.items():
        path = os.path.join(exp_path, exp)
        counts[exp] = {'counts': []}
        counts[exp]['name_exp'] = get_params_from_log(os.path.join(path, 'logs', 'flogger.log')) #To load the logger
        counts[exp]['save_path'] = os.path.join(path, 'analysis')
        for file in os.listdir(os.path.join(path, 'sample_counts')):
            if file.split('_')[-1].split('.')[0] in epochs:
                with open(os.path.join(path, 'sample_counts', file), 'rb') as f:
                    counts[exp]['counts'].append(pkl.load(f))
        if not os.path.isdir(counts[exp]['save_path']):
            os.makedirs(counts[exp]['save_path'], exist_ok=True)

    create_histograms(counts)
