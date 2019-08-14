import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os, json
import pdb
import matplotlib.ticker as ticker
import matplotlib as mpl


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
save_dir = '/home/matteo/Dropbox/results/nips/'
base = '/home/matteo/information_sampler/results/'
sns.set_color_codes()

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

def loadlog(f):
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
    return d

colors = ["windows blue", "red", "amber", "greyish", "faded green", "dusty purple"]
#sns.palplot(
#sns.xkcd_palette(colors)

def wrn1610():
    natural = '/home/matteo/information_sampler/results/(May_19_22_37_24)_opt_{"arch":"wrn1610","d":0.25,"dataset":"cifar10","wd":0.0005}'
    # robust = '/home/matteo/base/results/(Apr_12_19_50_41)_opt_{"arch":"wrn1610","d":0.25,"dataset":"cifar10","eps":8,"k":8,"wd":0.0005}'
    robust = '/home/matteo/information_sampler/results/(Apr_16_10_55_37)_opt_{"arch":"wrn1610","d":0.25,"dataset":"cifar10","k":8,"wd":0.0005}'
    return natural, robust, weigths_m


def plot_out_weights_distr(natural, robust, name):
    with open(natural+ '/weights_dir/weights_differences.pickle', 'rb') as h:
        wdn = pickle.load(h)
    with open(natural + '/weights_dir/weights_means.pickle', 'rb') as h:
        wmn = pickle.load(h)
    with open(robust+ '/weights_dir/weights_differences.pickle', 'rb') as h:
        wdr = pickle.load(h)
    with open(robust + '/weights_dir/weights_means.pickle', 'rb') as h:
        wmr = pickle.load(h)

    sns.distplot(wmn, hist_kws={'alpha': 0.4})
    sns.distplot(wmr, hist_kws={'alpha': 0.4})

    plt.legend(['CE','PGD'])
    plt.savefig(os.path.join(save_dir, name, 'wm.png'))
    plt.clf()
    plt.plot(wdn[:,0])
    plt.plot(wdr[:,0])
    plt.xlabel('Epoch')
    plt.ylabel(r"$|\hat{y}_t(x) - \hat{y}_{t-1}(x)|$")
    plt.clf()


def load_files(fs):
    # pkl = save_dir + '/wpgd.p'
    d = []
    # if os.path.isfile(pkl):
    #     return pickle.load(open(pkl, 'rb'))

    for f in fs:
        print(f)
        di = loadlog(f + '/logs/flogger.log')
        if 'gamma' not in di.columns:
            di['gamma'] = '0 (PGD)'
        d.append(di)

    d = pd.concat(d)
    #d.to_pickle(pkl, protocol=pickle.HIGHEST_PROTOCOL)
    return d


def compare_tiny_natural(train=False):
    set_size()
    mpl.rcParams['lines.linewidth'] = 2

    models = [
    base + '(Apr_25_20_22_26)_opt_{"arch":"wrn1610","d":0.25,"dataset":"tinyimagenet64","wd":0.0005}',
    base + '(Apr_27_18_10_03)_opt_{"arch":"wrn2810","d":0.25,"dataset":"tinyimagenet64","wd":0.0005}', 
    base + '(Apr_28_17_34_39)_opt_{"arch":"wrn4010","d":0.25,"dataset":"tinyimagenet64","wd":0.0005}'
    ]
   # return models
    df = load_files(models)
    # list models here
    #fs = ['']
    #get_params_from_log(f)
    whitelist = ['top1','loss','arch']
    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)
    #rename columns gamma -> p
    #dfc.rename(columns={'gamma': 'p'}, inplace=True)

    # only for wrn1610
    
    dv = dfc[(df['val'] == True)]
    # ax_top1 = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['top1'])
    # ax_loss = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['loss'])
    ax_top1 = sns.lineplot(x=dv.index/3, y='top1', hue='arch', data=dv)
    
    ax_top1.set(yscale="log")
    fm = ticker.ScalarFormatter()
    fm.set_scientific(False)
    ax_top1.yaxis.set_major_formatter(fm)
    ax_top1.yaxis.set_minor_formatter(fm)
    #ax_top1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1d'))
    plt.xlabel('Epochs')
    plt.ylabel('Natural Error [\%]')
    plt.savefig(os.path.join(save_dir + 'compare_tiny_natural_errors.pdf'), 
                bbox_inches='tight', format='pdf')
    
def compare_tiny_natural_PGD():
    set_size()
    mpl.rcParams['lines.linewidth'] = 2

    models = [
    base + '(Apr_25_20_22_26)_opt_{"arch":"wrn1610","d":0.25,"dataset":"tinyimagenet64","wd":0.0005}',
    base + '(May_04_10_53_11)_opt_{"arch":"wrn1610","dataset":"tinyimagenet64","k":8,"wd":0.0005}'
      ]
    # return models
    df = load_files(models)
    # list models here
    #fs = ['']
    #get_params_from_log(f)
    whitelist = ['top1','loss','k', 'e']
    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)
    #rename columns gamma -> p
    #dfc.rename(columns={'gamma': 'p'}, inplace=True)

    # only for wrn1610
    
    dv = dfc[(df['val'] == True)]
    dvc = dv.copy()
    dvc['s'] = 0
    dvc['s'][:200] = 0
    dvc['s'][200:] = 8
    dv = dvc
    import numpy as np
    # ax_top1 = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['top1'])
    # ax_loss = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['loss'])
    plt.clf()
    #sns.set_palette(sns.xkcd_palette(colors))
    #plt.style.use('ggplot')
    ax_top1 = sns.lineplot(x='e', y='top1', hue='s', data=dv, palette=sns.xkcd_palette(colors[:2]))
    ax_top1.set(yscale="log")
    fm = ticker.ScalarFormatter()
    fm.set_scientific(False)
    ax_top1.yaxis.set_major_formatter(fm)
    ax_top1.yaxis.set_minor_formatter(fm)
    #ax_top1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1d'))
    plt.xlabel('Epochs')
    plt.ylabel('Natural Error [\%]')
    plt.legend(['CE', 'PGD'])
    plt.savefig(os.path.join(save_dir + 'compare_tiny_natural_PGD_errors.pdf'), 
                bbox_inches='tight', format='pdf')


def compare_cifar10_natural_PGD_losses():
    set_size()

    mpl.rcParams['lines.linewidth'] = 2

    models = [
    '/home/matteo/base/results/' + '(Apr_12_19_50_41)_opt_{"arch":"wrn1610","d":0.25,"dataset":"cifar10","g":1,"wd":0.0005}',
    base + '(Apr_16_10_55_37)_opt_{"arch":"wrn1610","d":0.25,"dataset":"cifar10","k":8,"wd":0.0005}'
     ]
    # return models
    df = load_files(models)
    # list models here
    #fs = ['']
    #get_params_from_log(f)
    whitelist = ['top1','loss','k','e']


    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)
    #rename columns gamma -> p
    #dfc.rename(columns={'gamma': 'p'}, inplace=True)

    # only for wrn1610
    dv = dfc[(df['val'] == True)]

    dvc = dv.copy()
    dvc = dvc.reset_index()
    dvc['s'] = 0
    dvc['s'][:200] = 0
    dvc['s'][200:] = 8

    dv=dvc
    import numpy as np
    # ax_top1 = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['top1'])
    # ax_loss = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['loss'])
    plt.clf()
    #sns.set_palette(sns.xkcd_palette(colors))
    #plt.style.use('ggplot')
    ax_top1 = sns.lineplot(x='e', y='loss', hue='s', data=dv, palette=sns.xkcd_palette(colors[:2]))
    ax_top1.set(yscale="log")
    fm = ticker.ScalarFormatter()
    fm.set_scientific(False)
    ax_top1.yaxis.set_major_formatter(fm)
    ax_top1.yaxis.set_minor_formatter(fm)
    #ax_top1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1d'))
    plt.xlabel('Epochs')
    plt.ylabel('Natural Error [\%]')
    plt.legend(['CE', 'PGD'])
    plt.savefig(os.path.join(save_dir + 'compare_cifar10_natural_PGD_losses.pdf'), 
                bbox_inches='tight', format='pdf')


def compare_cifar10_natural_PGD():
    set_size()

    mpl.rcParams['lines.linewidth'] = 2
    models = [
    '/home/matteo/base/results/' + '(Apr_12_19_50_41)_opt_{"arch":"wrn1610","d":0.25,"dataset":"cifar10","g":1,"wd":0.0005}',
    base + '(Apr_16_10_55_37)_opt_{"arch":"wrn1610","d":0.25,"dataset":"cifar10","k":8,"wd":0.0005}'
     ]
    # return models
    df = load_files(models)
    # list models here
    #fs = ['']
    #get_params_from_log(f)
    whitelist = ['top1','loss','k','e']


    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)
    #rename columns gamma -> p
    #dfc.rename(columns={'gamma': 'p'}, inplace=True)

    # only for wrn1610
    dv = dfc[(df['val'] == True)]

    dvc = dv.copy()
    dvc = dvc.reset_index()
    dvc['s'] = 0
    dvc['s'][:200] = 0
    dvc['s'][200:] = 8

    dv=dvc
    import numpy as np
    # ax_top1 = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['top1'])
    # ax_loss = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['loss'])
    plt.clf()
    #sns.set_palette(sns.xkcd_palette(colors))
    #plt.style.use('ggplot')
    ax_top1 = sns.lineplot(x='e', y='top1', hue='s', data=dv, palette=sns.xkcd_palette(colors[:2]))
    ax_top1.set(yscale="log")
    fm = ticker.ScalarFormatter()
    fm.set_scientific(False)
    ax_top1.yaxis.set_major_formatter(fm)
    ax_top1.yaxis.set_minor_formatter(fm)
    #ax_top1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1d'))
    plt.xlabel('Epochs')
    plt.ylabel('Natural Error [\%]')
    plt.legend(['CE', 'PGD'])
    plt.savefig(os.path.join(save_dir + 'compare_cifar10_natural_PGD_errors.pdf'), 
                bbox_inches='tight', format='pdf')


# def compare_cifar10_wrn2810_natural_PGD():
#     models = [
#     base + '(Apr_24_18_04_10)_opt_{"arch":"wrn2810","dataset":"cifar10","k":8,"wd":0.0005}',
#      ]


def two_class_35():
    set_size()

    mpl.rcParams['lines.linewidth'] = 2
    models = [
        base + 'twocl_(May_18_16_36_45)_opt_{"arch":"wrn1610","classes":[3,5],"d":0.25,"dataset":"cifar10","epochs":140,"marker":"twocl","wd":0.0005}/',
        base + '(May_19_11_55_30)_opt_{"arch":"wrn1610","classes":[3,5],"d":0.25,"dataset":"cifar10","eps":8,"k":8,"wd":0.0005}/'
    ]
    df = load_files(models)
    # list models here
    #fs = ['']
    #get_params_from_log(f)
    whitelist = ['top1','loss','k','e']


    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)
    #rename columns gamma -> p
    #dfc.rename(columns={'gamma': 'p'}, inplace=True)

    # only for wrn1610
    dv = dfc[(df['val'] == True)]

    dvc = dv.copy()
    dvc = dvc.reset_index()
    dvc['s'] = 0
    dvc['s'][:140] = 0
    dvc['s'][140:] = 8

    dv=dvc
    import numpy as np
    # ax_top1 = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['top1'])
    # ax_loss = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['loss'])
    plt.clf()
    #sns.set_palette(sns.xkcd_palette(colors))
    #plt.style.use('ggplot')
    ax_top1 = sns.lineplot(x='e', y='top1', hue='s', data=dv, palette=sns.xkcd_palette(colors[:2]))
    ax_top1.set(yscale="log")
    fm = ticker.ScalarFormatter()
    fm.set_scientific(False)
    ax_top1.yaxis.set_major_formatter(fm)
    ax_top1.yaxis.set_minor_formatter(fm)
    #ax_top1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1d'))
    plt.xlabel('Epochs')
    plt.ylabel('Natural Error [\%]')
    plt.legend(['CE', 'PGD'])
    plt.savefig(os.path.join(save_dir + 'compare_cifar10_natural_35_PGD_errors.pdf'), 
                bbox_inches='tight', format='pdf')



def two_class_35_un():
    set_size()

    mpl.rcParams['lines.linewidth'] = 2
    models = [
        base + 'twocl_(May_21_19_17_42)_opt_{"arch":"wrn1610","b":64,"classes":[3,5],"d":0.25,"dataset":"cifar10","marker":"twocl","unbalanced":true,"wd":0.0005}/',
        base + 'twocl_(May_21_20_03_13)_opt_{"arch":"wrn1610","b":64,"classes":[3,5],"d":0.25,"dataset":"cifar10","eps":8,"k":8,"marker":"twocl","unbalanced":true,"wd":0.0005}/'
    ]
    df = load_files(models)
    # list models here
    #fs = ['']
    #get_params_from_log(f)
    whitelist = ['top1','loss','k','e']


    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)
    #rename columns gamma -> p
    #dfc.rename(columns={'gamma': 'p'}, inplace=True)

    # only for wrn1610
    dv = dfc[(df['val'] == True)]

    dvc = dv.copy()
    dvc = dvc.reset_index()
    dvc['s'] = 0
    dvc['s'][:200] = 0
    dvc['s'][200:] = 8

    dv=dvc
    import numpy as np
    # ax_top1 = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['top1'])
    # ax_loss = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['loss'])
    plt.clf()
    #sns.set_palette(sns.xkcd_palette(colors))
    #plt.style.use('ggplot')
    ax_top1 = sns.lineplot(x='e', y='top1', hue='s', data=dv, palette=sns.xkcd_palette(colors[:2]))
    ax_top1.set(yscale="log")
    fm = ticker.ScalarFormatter()
    fm.set_scientific(False)
    ax_top1.yaxis.set_major_formatter(fm)
    ax_top1.yaxis.set_minor_formatter(fm)
    #ax_top1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1d'))
    plt.xlabel('Epochs')
    plt.ylabel('Natural Error [\%]')
    plt.legend(['CE', 'PGD'])
    plt.savefig(os.path.join(save_dir + 'compare_cifar10_natural_un_35_PGD_errors.pdf'), 
                bbox_inches='tight', format='pdf')

def two_class_07_un():
    mpl.rcParams['lines.linewidth'] = 2
    set_size()

    models = [
        base + 'twocl_(May_21_19_44_26)_opt_{"arch":"wrn1610","b":64,"classes":[0,7],"d":0.25,"dataset":"cifar10","epochs":140,"marker":"twocl","unbalanced":true,"wd":0.0005}/',
        base + 'twocl_(May_21_22_11_04)_opt_{"arch":"wrn1610","b":64,"classes":[0,7],"d":0.25,"dataset":"cifar10","epochs":140,"eps":8,"k":8,"marker":"twocl","unbalanced":true,"wd":0.0005}/'
        ]

    df = load_files(models)
    # list models here
    #fs = ['']
    #get_params_from_log(f)
    whitelist = ['top1','loss','k','e']


    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)
    #rename columns gamma -> p
    #dfc.rename(columns={'gamma': 'p'}, inplace=True)

    # only for wrn1610
    dv = dfc[(df['val'] == True)]

    dvc = dv.copy()
    dvc = dvc.reset_index()
    dvc['s'] = 0
    dvc['s'][:140] = 0
    dvc['s'][140:] = 8

    dv=dvc
    import numpy as np
    # ax_top1 = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['top1'])
    # ax_loss = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['loss'])
    plt.clf()
    #sns.set_palette(sns.xkcd_palette(colors))
    #plt.style.use('ggplot')
    ax_top1 = sns.lineplot(x='e', y='top1', hue='s', data=dv, palette=sns.xkcd_palette(colors[:2]))
    ax_top1.set(yscale="log")
    fm = ticker.ScalarFormatter()
    fm.set_scientific(False)
    ax_top1.yaxis.set_major_formatter(fm)
    ax_top1.yaxis.set_minor_formatter(fm)
    #ax_top1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1d'))
    plt.xlabel('Epochs')
    plt.ylabel('Natural Error [\%]')
    plt.legend(['CE', 'PGD'])
    plt.savefig(os.path.join(save_dir + 'compare_cifar10_natural_07_un_PGD_errors.pdf'), 
                bbox_inches='tight', format='pdf')


def two_class_07():
    mpl.rcParams['lines.linewidth'] = 2
    set_size()

    models = [
        base + 'twocl_(May_18_17_10_49)_opt_{"arch":"wrn1610","classes":[0,7],"d":0.25,"dataset":"cifar10","epochs":140,"marker":"twocl","wd":0.0005}/',
        base + 'twocl_(May_18_20_53_06)_opt_{"arch":"wrn1610","classes":[0,7],"d":0.25,"dataset":"cifar10","epochs":140,"eps":8,"k":8,"marker":"twocl","wd":0.0005}/'
        ]
    df = load_files(models)
    # list models here
    #fs = ['']
    #get_params_from_log(f)
    whitelist = ['top1','loss','k','e']


    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)
    #rename columns gamma -> p
    #dfc.rename(columns={'gamma': 'p'}, inplace=True)

    # only for wrn1610
    dv = dfc[(df['val'] == True)]

    dvc = dv.copy()
    dvc = dvc.reset_index()
    dvc['s'] = 0
    dvc['s'][:140] = 0
    dvc['s'][140:] = 8

    dv=dvc
    import numpy as np
    # ax_top1 = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['top1'])
    # ax_loss = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['loss'])
    plt.clf()
    #sns.set_palette(sns.xkcd_palette(colors))
    #plt.style.use('ggplot')
    ax_top1 = sns.lineplot(x='e', y='top1', hue='s', data=dv, palette=sns.xkcd_palette(colors[:2]))
    ax_top1.set(yscale="log")
    fm = ticker.ScalarFormatter()
    fm.set_scientific(False)
    ax_top1.yaxis.set_major_formatter(fm)
    ax_top1.yaxis.set_minor_formatter(fm)
    #ax_top1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1d'))
    plt.xlabel('Epochs')
    plt.ylabel('Natural Error [\%]')
    plt.legend(['CE', 'PGD'])
    plt.savefig(os.path.join(save_dir + 'compare_cifar10_natural_07_PGD_errors.pdf'), 
                bbox_inches='tight', format='pdf')


def compare_cifar10_PGD():
    mpl.rcParams['lines.linewidth'] = 2
    set_size()

    models = [base + '(Apr_16_10_55_37)_opt_{"arch":"wrn1610","d":0.25,"dataset":"cifar10","k":8,"wd":0.0005}',
          base + '(Apr_24_18_04_10)_opt_{"arch":"wrn2810","dataset":"cifar10","k":8,"wd":0.0005}',
          base + '(Apr_16_11_00_20)_opt_{"arch":"wrn4010","d":0.25,"dataset":"cifar10","k":8,"wd":0.0005}'
          ]
    df = load_files(models)
    # list models here
    #fs = ['']
    #get_params_from_log(f)
    whitelist = ['top1','loss','arch','e']


    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)
    #rename columns gamma -> p
    #dfc.rename(columns={'gamma': 'p'}, inplace=True)

    # only for wrn1610
    dv = dfc[(df['val'] == True)]

    dvc = dv.copy()
    dvc = dvc.reset_index()
    
    dv=dvc
    import numpy as np
    # ax_top1 = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['top1'])
    # ax_loss = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['loss'])
    plt.clf()
    #sns.set_palette(sns.xkcd_palette(colors))
    #plt.style.use('ggplot')
    ax_top1 = sns.lineplot(x='e', y='top1', hue='arch', data=dv, palette=sns.xkcd_palette(colors[:3]))
    ax_top1.set(yscale="log")
    fm = ticker.ScalarFormatter()
    fm.set_scientific(False)
    ax_top1.yaxis.set_major_formatter(fm)
    ax_top1.yaxis.set_minor_formatter(fm)
    #ax_top1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1d'))
    plt.xlabel('Epochs')
    plt.ylabel('Natural Error [\%]')
    #plt.legend(['CE', 'PGD'])
    plt.savefig(os.path.join(save_dir + 'compare_cifar10_PGD_errors.pdf'), 
                bbox_inches='tight', format='pdf')
    
def compare_tiny_PGD():
    mpl.rcParams['lines.linewidth'] = 2
    set_size()

    models = [
         base +  '(May_04_10_53_11)_opt_{"arch":"wrn1610","dataset":"tinyimagenet64","k":8,"wd":0.0005}',
         base + '(May_05_23_30_54)_opt_{"arch":"wrn2810","dataset":"tinyimagenet64","eps":8,"k":8,"wd":0.0005}',
         ]          
    df = load_files(models)
    # list models here
    #fs = ['']
    #get_params_from_log(f)
    whitelist = ['top1','loss','arch','e']


    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)
    #rename columns gamma -> p
    #dfc.rename(columns={'gamma': 'p'}, inplace=True)

    # only for wrn1610
    dv = dfc[(df['val'] == True)]

    dvc = dv.copy()
    dvc = dvc.reset_index()
  

    dv=dvc
    import numpy as np
    # ax_top1 = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['top1'])
    # ax_loss = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['loss'])
    plt.clf()
    #sns.set_palette(sns.xkcd_palette(colors))
    #plt.style.use('ggplot')
    ax_top1 = sns.lineplot(x='e', y='top1', hue='arch', data=dv, palette=sns.xkcd_palette(colors[:2]))
    ax_top1.set(yscale="log")
    fm = ticker.ScalarFormatter()
    fm.set_scientific(False)
    ax_top1.yaxis.set_major_formatter(fm)
    ax_top1.yaxis.set_minor_formatter(fm)
    #ax_top1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1d'))
    plt.xlabel('Epochs')
    plt.ylabel('Natural Error [\%]')
    #plt.legend(['CE', 'PGD'])
    plt.savefig(os.path.join(save_dir + 'compare_tiny_natural_PGD_errors.pdf'), 
                bbox_inches='tight', format='pdf')

def wpgd_cifar10():
    mpl.rcParams['lines.linewidth'] = 2
    set_size()

    # only 28-10
    models = [
    '/home/matteo/base/results/' + '(May_13_14_09_36)_opt_{"arch":"wrn1610","d":0.25,"dataset":"cifar10","eps":8,"g":1,"k":8,"wd":0.0005}',
    base + 'wpgd_(May_17_00_48_20)_opt_{"arch":"wrn1610","d":0.25,"dataset":"cifar10","eps":8,"k":8,"wd":0.0005}',
    #base + 'wpgd_(May_17_23_03_56)_opt_{"arch":"wrn1610","d":0.25,"dataset":"cifar10","eps":8,"gamma":1.5,"k":8,"wd":0.0005}',
    base + 'wpgd_(May_17_00_48_20)_opt_{"arch":"wrn1610","d":0.25,"dataset":"cifar10","eps":8,"gamma":2.5,"k":8,"wd":0.0005}',
    base + 'wpgd_(May_20_11_11_02)_opt_{"arch":"wrn1610","d":0.25,"dataset":"cifar10","eps":8,"gamma":10.0,"k":8,"wd":0.0005}',
]
    df = load_files(models)
    # list models here
    #fs = ['']
    #get_params_from_log(f)
    df.rename(columns={'gamma': 'p'}, inplace=True)

    whitelist = ['top1','loss','p','e']


    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)
    
    # only for wrn1610
    dv = dfc[(df['val'] == True)]
    # ax_top1 = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['top1'])
    # ax_loss = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['loss'])
    plt.clf()
    #sns.set_palette(sns.xkcd_palette(colors))
    #plt.style.use('ggplot')
    ax_top1 = sns.lineplot(x='e', y='top1', hue='p', data=dv, palette=sns.xkcd_palette(colors[:4]))
    ax_top1.set(yscale="log")
    fm = ticker.ScalarFormatter()
    fm.set_scientific(False)
    ax_top1.yaxis.set_major_formatter(fm)
    ax_top1.yaxis.set_minor_formatter(fm)
    #ax_top1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1d'))
    plt.xlabel('Epochs')
    plt.ylabel('Natural Error [\%]')
    #plt.legend(['CE', 'PGD'])
    plt.savefig(os.path.join(save_dir + 'compare_cifar10_wpgd_errors.pdf'), 
                bbox_inches='tight', format='pdf')


def wpgd_tiny():
    mpl.rcParams['lines.linewidth'] = 2
    set_size()

    models = [
    base +  '(May_04_10_53_11)_opt_{"arch":"wrn1610","dataset":"tinyimagenet64","k":8,"wd":0.0005}',
    base + '(May_08_00_55_30)_opt_{"arch":"wrn1610","dataset":"tinyimagenet64","eps":8,"k":8,"wd":0.0005}',
    base + 'wpgd_(May_16_16_54_54)_opt_{"arch":"wrn1610","d":0.25,"dataset":"tinyimagenet64","eps":8,"gamma":2.5,"k":8,"wd":0.0005}',
    base + 'wpgd_(May_21_02_38_15)_opt_{"arch":"wrn1610","d":0.25,"dataset":"tinyimagenet64","eps":8,"gamma":10.0,"k":8,"wd":0.0005}',
    #'(May_05_23_30_54)_opt_{"arch":"wrn2810","dataset":"tinyimagenet64","eps":8,"k":8,"wd":0.0005}',
    ]
    df = load_files(models)
    # list models here
    #fs = ['']
    #get_params_from_log(f)
    df.rename(columns={'gamma': 'p'}, inplace=True)

    whitelist = ['top1','loss','p','e']


    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)
    
    # only for wrn1610
    dv = dfc[(df['val'] == True)]
    # ax_top1 = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['top1'])
    # ax_loss = sns.lineplot(x="epoch", y="Validation Error [$\%$]", hue='p', y=dv['loss'])
    plt.clf()
    #sns.set_palette(sns.xkcd_palette(colors))
    #plt.style.use('ggplot')
    ax_top1 = sns.lineplot(x='e', y='top1', hue='p', data=dv, palette=sns.xkcd_palette(colors[:4]))
    ax_top1.set(yscale="log")
    fm = ticker.ScalarFormatter()
    fm.set_scientific(False)
    ax_top1.yaxis.set_major_formatter(fm)
    ax_top1.yaxis.set_minor_formatter(fm)
    #ax_top1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1d'))
    plt.xlabel('Epochs')
    plt.ylabel('Natural Error [\%]')
    #plt.legend(['CE', 'PGD'])
    plt.savefig(os.path.join(save_dir + 'compare_tiny_wpgd_errors.pdf'), 
                bbox_inches='tight', format='pdf')


# def wpgd_tiny_full():
#     models = [
#     base +  '(May_04_10_53_11)_opt_{"arch":"wrn1610","dataset":"tinyimagenet64","k":8,"wd":0.0005}',
#     base + '(May_08_00_55_30)_opt_{"arch":"wrn1610","dataset":"tinyimagenet64","eps":8,"k":8,"wd":0.0005}',
#     base + 'wpgd_(May_16_16_54_54)_opt_{"arch":"wrn1610","d":0.25,"dataset":"tinyimagenet64","eps":8,"gamma":2.5,"k":8,"wd":0.0005}',
#     base + 'wpgd_(May_21_02_38_15)_opt_{"arch":"wrn1610","d":0.25,"dataset":"tinyimagenet64","eps":8,"gamma":10.0,"k":8,"wd":0.0005}',
#     base +'(Apr_27_18_10_03)_opt_{"arch":"wrn2810","d":0.25,"dataset":"tinyimagenet64","wd":0.0005}',
#     base +'(May_05_23_30_54)_opt_{"arch":"wrn2810","dataset":"tinyimagenet64","eps":8,"k":8,"wd":0.0005}',
#     base +'(Apr_28_17_34_39)_opt_{"arch":"wrn4010","d":0.25,"dataset":"tinyimagenet64","wd":0.0005}'
#     ]
#     return models


def main():
    #ax_top1 = pgd_losses_errors(compare_cifar10_PGD())
    compare_tiny_natural()
    two_class_07_un()
    two_class_35_un()
    two_class_35()
    two_class_07()
    compare_cifar10_natural_PGD_losses()
    compare_tiny_natural_PGD()
    compare_cifar10_natural_PGD()
    compare_cifar10_PGD()
    wpgd_cifar10()
    wpgd_tiny()
    # ax_top1.savefig(os.path.join(save_dir + 'compare_tiny_natural.pdf'), 
    #             bbox_inches='tight', format='pdf')

main()
