import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os, json
import pdb
import matplotlib.ticker as ticker
import glob2
import itertools


sns.set('paper')
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.set_color_codes()
colors = ["windows blue", "red", "amber", "greyish", "faded green", "dusty purple"]

blacklist = ['lrs', 'B', 'b', 'd', 's', 'ids', 'd', \
            'save', 'metric', 'nc', 'g', 'j', 'env', 'burnin', \
            'alpha', 'delta', 'beta', 'lambdas', 'append', \
            'ratio', 'bfrac', 'l2', 'v', 'frac', 'rhos']

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

def get_params_from_log(f):
    r = {}
    for l in open(f,'r', encoding='latin1'):
        if '[OPT]' in l[:5]:
            r = json.loads(l[5:-1])
            fn = r['filename']
            for k in blacklist:
                r.pop(k, None)
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

def get_default_value(name):
    def_values = {'arch':'resnet10','dataset':'cifar10','b':128,'adam':False,'sampler':'default','normalizer':False,
                  'temperature':1.,'modatt':False,'dyncount':False,'adjust_classes':False,'ac_scaler':10,'bce':False,
                  'use_train_clean':False,'corr_labels':0.}
    return def_values[name]

def get_data_loss_top1(opt):
    # list of source directories
    source_dir = ['{}{}_{}_{}'.format(opt['base'],opt['exp'],pair[0],pair[1]) for pair in list(itertools.product(opt['arch'], opt['datasets']))]
    print(source_dir)
    print()
    d = []
    for dir_ in source_dir:
        expts = glob2.glob(f'{dir_}/**/*.log')
        ex = []
        # remove pilots
        for f in expts:
            fstr = '{' + f.split('{')[1].split('}')[0] + '}'
            keywords = json.loads(fstr)
            if 'pilot' not in keywords.keys():
                ex.append(f)

        for f in ex:
            fstr = '{' + f.split('{')[1].split('}')[0] + '}'
            keywords = json.loads(fstr)
            # default settings
            kws = dict()
            if 'arch' not in keywords.keys():
                keywords['arch'] = 'resnet10'
                kws['arch'] = keywords['arch']
            if 'dataset' not in keywords.keys():
                keywords['dataset'] = 'cifar10'
                kws['dataset'] = keywords['dataset']
            if 'use_train_clean' not in keywords.keys():
                keywords['use_train_clean'] = False
                kws['use_train_clean'] = keywords['use_train_clean']
            if opt['hue'] is not None and opt['hue'] not in keywords.keys():
                keywords[opt['hue']] = get_default_value(opt['hue'])
                kws[opt['hue']] = keywords[opt['hue']]
            if opt['style'] is not None and opt['style'] not in keywords.keys():
                keywords[opt['style']] = get_default_value(opt['style'])
                kws[opt['style']] = keywords[opt['style']]
            if opt['size'] is not None and opt['size'] not in keywords.keys():
                keywords[opt['size']] = get_default_value(opt['size'])
                kws[opt['size']] = keywords[opt['size']]
            if opt['cfl'] is not None:
                for name in opt['cfl']:
                    if name not in keywords.keys():
                        keywords[name] = get_default_value(name)
                        kws[name] = keywords[name]
            # load data
            di = loadlog(f, kws=kws)
            d.append(di)

    df = pd.concat(d)
    whitelist = ['use_train_clean','train_clean','train','val','arch','dataset','e','loss', 'top1']
    if opt['cfl'] is not None:
        for param in opt['cfl']:
            whitelist.append(param)
    for param in [opt['hue'],opt['style'],opt['size']]:
        if param is not None:
            whitelist.append(param)
    # remove duplicates
    whitelist = list(set(whitelist))

    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)

    if dfc.use_train_clean.nunique()>1: 
        # we used train_clean
        df_train = dfc[dfc.train_clean==True]
    else:
        # we did not use train_clean 
        df_train = dfc[dfc.train==True]
    df_val = dfc[dfc.val==True]

    df_train['phase'] = ['train' for i in range(len(df_train))]
    df_val['phase'] = ['val' for i in range(len(df_val))]

    return pd.concat([df_train, df_val])

def get_filename(keys, values, opt):
    # name of the "comparison at plot level" part
    cpl = [opt['hue'], opt['style'], opt['size']]
    fn_cpl = '{'+','.join(item for item in cpl if item)+'}'
    # name of the "comparison at file level" part
    fn_cfl = []
    for i in range(len(keys)):
        fn_cfl.append(':'.join([keys[i],str(values[i])]))
    fn_cfl = '{'+','.join(fn_cfl)+'}'
    # complete name
    fn = fn_cpl+'_'+fn_cfl
    return fn

def plot(opt):
    # load data
    df = get_data_loss_top1(opt)
    # filter by phase (train or val)
    df = df[df.phase==opt['phase']]
    # create dict with values assumed by variables in opt['cfl']
    cfl = {}
    if opt['cfl'] is not None:
        for el in opt['cfl']:
            cfl[el] = df[el].unique()
    # create a plot for every combination of values assumed by variables in opt['cfl'] (comparisons at file level)
    keys = list(cfl.keys())
    for values in list(itertools.product(*cfl.values())):
        dfc = df.copy()
        for i in range(len(keys)):
            dfc = dfc[dfc[keys[i]]==values[i]]
        plt.clf()
        ax = sns.lineplot(x='e', y=opt['plot'], hue=opt['hue'], style=opt['style'], size=opt['size'], data=dfc)
        fm = ticker.ScalarFormatter()
        fm.set_scientific(False)
        ax.yaxis.set_major_formatter(fm)
        ax.yaxis.set_minor_formatter(fm)
        ax.set(xlabel='epoch', ylabel=opt['phase'] + ' ' + opt['plot'])
        dir_ = os.path.join(opt['sd'], opt['exp'])
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        fn = get_filename(keys, values, opt)
        plt.savefig(os.path.join(dir_, opt['plot'] + '_' + opt['phase'] + '_' + fn + '.pdf'), bbox_inches='tight', format='pdf')
        plt.close()

def plot_train_val(opt):
    # load data
    df = get_data_loss_top1(opt)
    # create dict with values assumed by variables in opt['cfl']
    cfl = {}
    if opt['cfl'] is not None:
        for el in opt['cfl']:
            cfl[el] = df[el].unique()
    # create a plot for every combination of values assumed by variables in opt['cfl'] (comparisons at file level)
    keys = list(cfl.keys())
    for values in list(itertools.product(*cfl.values())):
        dfc = df.copy()
        for i in range(len(keys)):
            dfc = dfc[dfc[keys[i]]==values[i]]
        plt.clf()
        ax = sns.lineplot(x='e', y=opt['plot'], hue=opt['hue'], style='phase', size=opt['size'], data=dfc)
        fm = ticker.ScalarFormatter()
        fm.set_scientific(False)
        ax.yaxis.set_major_formatter(fm)
        ax.yaxis.set_minor_formatter(fm)
        ax.set(xlabel='epoch', ylabel=opt['plot'])
        dir_ = os.path.join(opt['sd'], opt['exp'])
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        fn = get_filename(keys, values, opt)
        plt.savefig(os.path.join(dir_, opt['plot'] + '_train_val_' + fn + '.pdf'), bbox_inches='tight', format='pdf')
        plt.close()

