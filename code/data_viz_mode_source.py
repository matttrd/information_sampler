import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os, json
import pdb
import matplotlib.ticker as ticker
import matplotlib as mpl
import glob2, argparse
from IPython import embed
from os.path import expanduser
home = expanduser("~")

def map_mode(x):
    int2str = {0: 'most diff', 1: 'easiest', 2: 'random',
               3: 'farthest', 4: 'nearest', 5: 'random'}
    return int2str[x]


def set_size():
    fsz = 20
    plt.rc('text', usetex=True)
    plt.rc('font', size=fsz)
    plt.rc('axes', titlesize=fsz)
    plt.rc('axes', labelsize=fsz)
    plt.rc('xtick', labelsize=fsz)
    plt.rc('ytick', labelsize=fsz)
    plt.rc('legend', fontsize=fsz)
    plt.rc('figure', titlesize=fsz)

sns.set('paper')
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]

parser = argparse.ArgumentParser(description='data viz',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--sd', default=f'{home}/Dropbox/results/')
parser.add_argument('--base', default='../results')
#parser.add_argument('--exp', default='', help='marker for filename')
opt = vars(parser.parse_args())
sns.set_color_codes()

save_dir = opt['sd']
base = opt['base']

blacklist = ['lrs', 'B', 'b', 'd', 's', 'ids', 'd', \
            'save', 'metric', 'nc', 'g', 'j', 'env', 'burnin', \
            'alpha', 'delta', 'beta', 'lambdas', 'append', \
            'ratio', 'bfrac', 'l2', 'v', 'frac', 'rhos']

def get_params_from_log(f):
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

def loadlog(f, kws=None):
    logs, summary = [], []
    opt = get_params_from_log(f)
    for l in open(f):
        if '[LOG]' in l[:5]:
            logs.append(json.loads(l[5:-1]))
        elif '[SUMMARY]' in l[:9]:
            try:
                summary.append(json.loads(l[9:-1]))
            except ValueError:
                pdb.set_trace()
        else:
            try:
                s = json.loads(l)
            except:
                continue
            if s['i'] == 0:
                if not 'val' in s:
                    s['train'] = True
                summary.append(s)
            else:
                logs.append(s)
    dl, ds = pd.DataFrame(logs), pd.DataFrame(summary)

    dl['log'] = True
    ds['summary'] = True
    for k in opt:
        try:
            dl[k] = opt[k]
            ds[k] = opt[k]
        except Exception:
            pass
  
    d = pd.concat([dl, ds])
    if kws is not None:
        for k,v in kws.items():
            d[k] = v

    return d

colors = ["windows blue", "red", "amber", "greyish", "faded green", "dusty purple"]
#sns.palplot(
#sns.xkcd_palette(colors)

def load_files(fs):
    # pkl = save_dir + '/wpgd.p'
    d = []
    # if os.path.isfile(pkl):
    #     return pickle.load(open(pkl, 'rb'))

    for f in fs:
        print(f)
        di = loadlog(f)
        d.append(di)

    d = pd.concat(d)
    #d.to_pickle(pkl, protocol=pickle.HIGHEST_PROTOCOL)
    return d

# TODO: remember to modify this when creating EXP folder
def plot_MD_exp():
    expts = glob2.glob(f'{base}/**/*.log')
    d = []
    ex = []
    # remove pilots
    for f in expts:
        if f.split('/')[-3] == 'analysis_experiments':
                continue
        fstr = '{' + f.split('{')[1].split('}')[0] + '}'
        keywords = json.loads(fstr)
        if 'pilot' not in keywords.keys():
            ex.append(f)
    
    for f in ex:
        #find it in the file name
        fstr = '{' + f.split('{')[1].split('}')[0] + '}'
        keywords = json.loads(fstr)
        
        if 'mode_source' not in keywords.keys():
            keywords['mode_source'] = 'counter'
        if 'perc' not in keywords.keys():
            keywords['perc'] = 0

        kws = dict()
        kws['mode_source'] = keywords['mode_source']
        kws['perc'] = keywords['perc']

        di = loadlog(f, kws=kws)
        d.append(di)
    
    df = pd.concat(d)
    whitelist = ['top1','perc','mode_source', 'val', 'e']
    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)
    dv = dfc[(dfc['val'] == True)]
    dv = dv[dv['e'] == 179]
    embed()
    plt.clf()
    ax_top1 = sns.lineplot(x='perc', y='top1', hue='mode_source', data=dv)
    bn = opt['base'].split('/')[-1]
    #ax_top1.set(yscale="log")
    fm = ticker.ScalarFormatter()
    fm.set_scientific(False)
    ax_top1.yaxis.set_major_formatter(fm)
    ax_top1.yaxis.set_minor_formatter(fm)
    ax_top1.set(xlabel='Perc removed', ylabel='Top1 acc')
    plt.savefig(os.path.join(save_dir + bn + '.pdf'), 
                bbox_inches='tight', format='pdf')


def main():
    plot_MD_exp()

main()