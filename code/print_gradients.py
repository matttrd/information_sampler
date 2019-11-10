from IPython import embed
import argparse
import os
from count_visualization import kde_sklearn, get_params_from_log
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

parser = argparse.ArgumentParser(description='gradients_stats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp', default=f'..{os.sep}results', required=True)
opt = vars(parser.parse_args())

# Returns the value associated to the key in a cerain string
def string_splitter(string, key):
    splitted = string.split('_')
    for i, term in enumerate(splitted):
        if key == term:
            return splitted[i+1]
    raise IndexError('string_spitter is not able to find you key!!!')


# For each run creates grad stats dictionary divided in IS and default
def load_grad_stats(run):
    import collections

    curr_path = os.path.join(run, 'gradients_stats')

    # Find the number of batches and create the dictionary
    dict_stats = dict()
    for stats in os.listdir(curr_path):
        if not string_splitter(stats, 'BS') in dict_stats:
            dict_stats[string_splitter(stats, 'BS')] = {}
        if not stats.split('_')[0] in dict_stats[string_splitter(stats, 'BS')]:
            dict_stats[string_splitter(stats, 'BS')][stats.split('_')[0]]= [stats]
        else:
            dict_stats[string_splitter(stats, 'BS')][stats.split('_')[0]].append(stats)

    # Sorting by BS
    dict_stats = collections.OrderedDict(sorted(dict_stats.items()))

    # Ordering by sampler
    for bs in dict_stats:
        dict_stats[bs] = collections.OrderedDict(sorted(dict_stats[bs].items()))

    #Sortig
    for bs in dict_stats:
        for sampler in dict_stats[bs]:
            dict_stats[bs][sampler] = sorted(dict_stats[bs][sampler], key=lambda x: int(x.split('_')[-1].split('.')[0]))


    return dict_stats

exp_path = os.path.join('..', 'results', opt['exp'])

runs = {}
for run in os.listdir(exp_path):
    if not 'analysis' in run:
        runs[run] = {}
        runs[run]['info_run'] = get_params_from_log(os.path.join(exp_path, run, 'logs', 'flogger.log')) #To load the logger
        runs[run]['dict_stats'] = load_grad_stats(os.path.join(exp_path, run))

# Print histograms of gradients for col: (BS, Importance sampling, default) // rows: epochs
def load_cosine_directors(exp_path, run, epoch_model):
    path = os.path.join(exp_path, run,'gradients_stats', epoch_model)
    with open(path, 'rb') as file:
        data = pkl.load(file)

    return data['cosine_directors']

def load_gradients_norms_directors(exp_path, run, epoch_model):
    path = os.path.join(exp_path, run,'gradients_stats', epoch_model)
    with open(path, 'rb') as file:
        data = pkl.load(file)

    return data['sum_mean_grad'], data['sum_var_grad'], data['gradients_norm']


# Comparing histograms default vs importance sampling
for run in runs.keys():
    current = runs[run]['dict_stats']
    fig, axs = plt.subplots(len(current.keys()), len(current[list(current.keys())[0]]['default']))
    info = runs[run]['info_run']
    title = info['sampler'] + '_' + 'Temp' + '_' + str(info['temperature']).replace('.', '_') + '_' + 'norm_' + str(info['normalizer'])
    fig.set_size_inches(20.5, 10.5)
    z = 0
    for i, bs in enumerate(current.keys()):
        for j, sampler in enumerate(current[bs]):
            for k, epoch_model in enumerate(current[bs][sampler]):
                cosine_directors = load_cosine_directors(exp_path, run, epoch_model)
                if max(cosine_directors) > 1 or  min(cosine_directors) < -1:
                    bins = np.linspace(min(cosine_directors),max(cosine_directors),100)
                else:
                    bins = np.linspace(0,1,100)
                #If due to matplotlib and special case of subplots
                if len(current.keys()) == 1:
                    axs[k].hist(cosine_directors, alpha=0.5, bins=bins, label=sampler)
                    axs[k].set_title(f"Bs {bs} Epoch {string_splitter(epoch_model, 'model').split('.')[0]} ")
                else:
                    axs[z][k].hist(cosine_directors, alpha=0.5, bins=bins, label=sampler)
                    axs[z][k].set_title(f"Bs {bs} Epoch {string_splitter(epoch_model, 'model').split('.')[0]} ")
                    axs[z][k].grid()
                    axs[z][k].legend()
        z += 1
    fig.suptitle(title, fontsize=30), plt.tight_layout()
    saving_path = os.path.join(exp_path, run, 'analysis')
    os.makedirs(saving_path, exist_ok=True)
    plt.savefig(os.path.join(saving_path, f'gradients_hists_{title}.pdf'), dpi=2560, bbox_inches='tight')
    plt.close()
    # plt.show()

# Comparing cosines angles at differnt epochs
for run in runs.keys():
    current = runs[run]['dict_stats']
    info = runs[run]['info_run']
    title = info['sampler'] + '_' + 'Temp' + '_' + str(info['temperature']).replace('.', '_') + '_' + 'norm_' + str(info['normalizer'])
    for i, bs in enumerate(current.keys()):
        for j, sampler in enumerate(current[bs]):
            plt.figure()
            fig.set_size_inches(20.5, 10.5)
            for k, epoch_model in enumerate(current[bs][sampler]):
                cosine_directors = load_cosine_directors(exp_path, run, epoch_model)
                if max(cosine_directors) > 1 or  min(cosine_directors) < -1:
                    bins = np.linspace(min(cosine_directors),max(cosine_directors),100)
                else:
                    bins = np.linspace(0,1,100)
                    plt.hist(cosine_directors, alpha=0.5, bins=bins, label=f"Epoch {string_splitter(epoch_model, 'model').split('.')[0]}")
            plt.grid()
            plt.title(title), plt.tight_layout()
            plt.legend()
            saving_path = os.path.join(exp_path, run, 'analysis')
            plt.savefig(os.path.join(saving_path, f'gradients_hists_epochs_BS_{bs}_{title}.pdf'), dpi=2560, bbox_inches='tight')
            plt.close()

# Comparing cosines angles at differnt epochs using kde
for run in runs.keys():
    current = runs[run]['dict_stats']
    info = runs[run]['info_run']
    title = info['sampler'] + '_' + 'Temp' + '_' + str(info['temperature']).replace('.', '_') + '_' + 'norm_' + str(info['normalizer'])
    for i, bs in enumerate(current.keys()):
        for j, sampler in enumerate(current[bs]):
            plt.figure()
            fig.set_size_inches(20.5, 10.5)
            for k, epoch_model in enumerate(current[bs][sampler]):
                cosine_directors = load_cosine_directors(exp_path, run, epoch_model)
                if max(cosine_directors) > 1 or  min(cosine_directors) < -1:
                    bins = np.linspace(min(cosine_directors),max(cosine_directors),100)
                else:
                    bins = np.linspace(0,1,100)
                    pdf = kde_sklearn(cosine_directors, bins, bandwidth=0.02)
                    plt.plot(bins, pdf, label=f"Epoch {string_splitter(epoch_model, 'model').split('.')[0]}", linewidth=2)
            plt.grid()
            plt.title(title), plt.tight_layout()
            plt.legend()
            saving_path = os.path.join(exp_path, run, 'analysis')
            plt.savefig(os.path.join(saving_path, f'gradients_kde_epochs_BS_{bs}_{title}.pdf'), dpi=2560, bbox_inches='tight')
            plt.close()

# Comparing cosines angles at differnt epochs using kde
for run in runs.keys():
    current = runs[run]['dict_stats']
    info = runs[run]['info_run']
    title = info['sampler'] + '_' + 'Temp' + '_' + str(info['temperature']).replace('.', '_') + '_' + 'norm_' + str(info['normalizer'])
    for i, bs in enumerate(current.keys()):
        for j, sampler in enumerate(current[bs]):
            plt.figure()
            fig.set_size_inches(20.5, 10.5)
            for k, epoch_model in enumerate(current[bs][sampler]):
                sum_mean_grad, sum_var_grad, grad_norms = load_gradients_norms_directors(exp_path, run, epoch_model)
                plt.hist(grad_norms, alpha=0.5, bins=100, label=f"Epoch {string_splitter(epoch_model, 'model').split('.')[0]}")

            plt.grid()
            plt.title(title + f" sum_mean_g: {sum_mean_grad:.3f} sum_var_g: {sum_var_grad:.3f}"), plt.tight_layout()
            plt.legend()
            saving_path = os.path.join(exp_path, run, 'analysis')
            plt.savefig(os.path.join(saving_path, f'norm_gradients_hist_epochs_BS_{bs}_{title}.pdf'), dpi=2560, bbox_inches='tight')
            plt.close()
            # plt.show()

# Comparing cosines angles at differnt epochs using kde
for run in runs.keys():
    current = runs[run]['dict_stats']
    info = runs[run]['info_run']
    title = info['sampler'] + '_' + 'Temp' + '_' + str(info['temperature']).replace('.', '_') + '_' + 'norm_' + str(info['normalizer'])
    for i, bs in enumerate(current.keys()):
        for j, sampler in enumerate(current[bs]):
            plt.figure()
            fig.set_size_inches(20.5, 10.5)
            for k, epoch_model in enumerate(current[bs][sampler]):
                sum_mean_grad, sum_var_grad, grad_norms = load_gradients_norms_directors(exp_path, run, epoch_model)

                bins = np.linspace(min(grad_norms)/1.2,max(grad_norms) * 1.2 ,100)
                pdf = kde_sklearn(grad_norms, bins, bandwidth=0.05)
                plt.plot(bins, pdf, label=f"Epoch {string_splitter(epoch_model, 'model').split('.')[0]}", linewidth=2)
            plt.grid()
            plt.title(title + f" sum_mean_g: {sum_mean_grad:.3f} sum_var_g: {sum_var_grad:.3f}"), plt.tight_layout()
            plt.legend()
            saving_path = os.path.join(exp_path, run, 'analysis')
            plt.savefig(os.path.join(saving_path, f'norm_gradients_kde_epochs_BS_{bs}_{title}.pdf'), dpi=2560, bbox_inches='tight')
            plt.close()
