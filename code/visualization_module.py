import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os, json
import pdb
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import glob2
import itertools
import numpy as np
import re
import pickle

sns.set('paper')
sns.set_color_codes()
colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown",
          "tab:pink","tab:gray","tab:olive","tab:cyan"] 
#colors = ['b','r','g','c']

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

def check_default_compared(opt, keywords):
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
    if opt['hue'] is not None and list(opt['hue'].keys())[0] not in keywords.keys():
        keywords[list(opt['hue'].keys())[0]] = get_default_value(list(opt['hue'].keys())[0])
        kws[list(opt['hue'].keys())[0]] = keywords[list(opt['hue'].keys())[0]]
    if opt['style'] is not None and list(opt['style'].keys())[0] not in keywords.keys():
        keywords[list(opt['style'].keys())[0]] = get_default_value(list(opt['style'].keys())[0])
        kws[list(opt['style'].keys())[0]] = keywords[list(opt['style'].keys())[0]]
    if opt['size'] is not None and list(opt['size'].keys())[0] not in keywords.keys():
        keywords[list(opt['size'].keys())[0]] = get_default_value(list(opt['size'].keys())[0])
        kws[list(opt['size'].keys())[0]] = keywords[list(opt['size'].keys())[0]]
    if opt['cfl'] is not None:
        for name in opt['cfl'].keys():
            if name not in keywords.keys():
                keywords[name] = get_default_value(name)
                kws[name] = keywords[name]
    return kws

def check_default_single(opt, keywords):
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
    
    return kws


def get_all_data_loss_top1(opt):
    # list of source directories
    source_dir = ['{}{}_{}_{}'.format(opt['base'],opt['exp'],pair[0],pair[1]) for pair in list(itertools.product(opt['arch'], opt['datasets']))]
    d = []
    for dir_ in source_dir:
        expts = glob2.glob(f'{dir_}/**/*.log')
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
            fstr = '{' + f.split('{')[1].split('}')[0] + '}'
            keywords = json.loads(fstr)
            # default settings
            kws = check_default_compared(opt, keywords)
            # load data
            di = loadlog(f, kws=kws)
            d.append(di)
    df = pd.concat(d)
    whitelist = ['use_train_clean','train_clean','train','val','arch','dataset','e','loss', 'top1']
    if opt['cfl'] is not None:
        for param in opt['cfl'].keys():
            whitelist.append(param)
    for param in [opt['hue'],opt['style'],opt['size']]:
        if param is not None:
            whitelist.append(list(param.keys())[0])
    # remove duplicates
    whitelist = list(set(whitelist))

    df = df[(df['summary'] == True)]
    dfc = df.copy()
    dfc = dfc.filter(items=whitelist)

    if all(dfc.use_train_clean.unique()): 
        # we used train_clean
        df_train = dfc[dfc.train_clean==True]
    else:
        # we did not use train_clean 
        df_train = dfc[dfc.train==True]
    df_val = dfc[dfc.val==True]

    df_train['phase'] = ['train' for i in range(len(df_train))]
    df_val['phase'] = ['val' for i in range(len(df_val))]

    return pd.concat([df_train, df_val])

def single_plot_train_val(opt):
    exp_dir = opt['exp_dir']
    expts = glob2.glob(f'{exp_dir}/**/*.log')
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
        run_name = '/'.join(f.split('/')[:-2])
        fstr = '{' + f.split('{')[1].split('}')[0] + '}'
        keywords = json.loads(fstr)
        # default settings
        kws = check_default_single(opt, keywords)
        # load data
        df = loadlog(f, kws=kws)
        whitelist = ['use_train_clean','train_clean','train','val','e','loss', 'top1']
        df = df[(df['summary'] == True)]
        dfc = df.copy()
        dfc = dfc.filter(items=whitelist)
        # train-val
        if all(dfc.use_train_clean.unique()): 
            # we used train_clean
            df_train = dfc[dfc.train_clean==True]
        else:
            # we did not use train_clean 
            df_train = dfc[dfc.train==True]
        df_val = dfc[dfc.val==True]
        df_train['phase'] = ['train' for i in range(len(df_train))]
        df_val['phase'] = ['val' for i in range(len(df_val))]
        d = pd.concat([df_train, df_val])
        # create plot
        ax = sns.lineplot(x='e', y=opt['plot'], style='phase', data=d)
        fm = ticker.ScalarFormatter()
        fm.set_scientific(False)
        ax.yaxis.set_major_formatter(fm)
        ax.yaxis.set_minor_formatter(fm)
        ax.set(xlabel='epoch', ylabel=opt['plot'])
        # save plot
        dir_ = os.path.join(run_name, 'analysis')
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        plt.savefig(os.path.join(dir_, opt['plot'] + '.pdf'), bbox_inches='tight', format='pdf')
        plt.close()


def get_all_data_counts(opt):
    # list of source directories
    source_dir = ['{}{}_{}_{}'.format(opt['base'],opt['exp'],pair[0],pair[1]) for pair in list(itertools.product(opt['arch'], opt['datasets']))]
    d = []
    max_count = 0
    for dir_ in source_dir:
        expts = glob2.glob(f'{dir_}/**/sample_counts')
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
            kws = check_default_compared(opt, keywords)
            # load data
            train_labels = np.load('./train_labels_' + keywords['dataset'] + '.npy')
            num_classes = len(np.unique(train_labels))
            data = []
            for e in opt['epochs']:
                with open(os.path.join(f,'sample_counts_' + str(e) + '.pkl'), 'rb') as file:
                    counts = pickle.load(file)
                class_count = np.zeros(10)
                for sample_id in range(counts.shape[0]):
                    current_class = train_labels[sample_id]
                    class_count[current_class] += counts[sample_id]
                total_count = np.sum(class_count)
                #class_count = class_count / total_count
                data.append(class_count)
            keywords['data'] = data
            di = pd.DataFrame(keywords)
            d.append(di)
            current_max_count = np.max(data[-1])
            if current_max_count > max_count:
                max_count = current_max_count
    df = pd.concat(d)  

    return df, max_count, num_classes

def get_filename(keys, values, opt):
    # name of the "comparison at plot level" part
    cpl = [opt['hue'], opt['style'], opt['size']]
    fn_cpl = '{'+','.join(list(item.keys())[0] for item in cpl if item)+'}'
    # name of the "comparison at file level" part
    fn_cfl = []
    for i in range(len(keys)):
        fn_cfl.append(':'.join([keys[i],str(values[i])]))
    fn_cfl = '{'+','.join(fn_cfl)+'}'
    # complete name
    fn = fn_cpl+'_'+fn_cfl
    return fn

def get_filename_clean(keys, values, opt):
    fn = []
    for i in range(len(keys)):
        fn.append(':'.join([keys[i],str(values[i])]))
    fn = '__'.join(fn)
    fn = fn.replace('normalizer:True','normalizer')
    fn = fn.replace('normalizer:False','')
    fn = fn.replace(':','_')
    for word in ['sampler_','arch_','dataset_']:
        fn = fn.replace(word,'')
    fn = fn.replace('temperature', 'temp')
    fn = fn.replace('.', 'point')
    while fn.endswith('_'):
        fn = fn[:-2]
    return fn

def compared_plot(opt):
    options = [opt[name] for name in ['hue','style','size'] if opt[name]]
    plot_opt = []
    for name in ['hue','style','size']:
        if opt[name]:
            plot_opt.append(list(opt[name].keys())[0])
        else:
            plot_opt.append(opt[name])
    # load data
    df = get_all_data_loss_top1(opt)
    # filter by phase (train or val)
    df = df[df.phase==opt['phase']]
    # consider only specified values for opt['hue'], opt['size'], opt['size']
    for option in options:
        key = list(option.keys())[0]
        if option[key] == 'all':
            # use all values and do not filter
            print('Keyword {}: using all values'.format(key))
        else:
            # filter out non-selected values
            selected_values = []
            for elem in option[key]:
                selected_values.append(elem)
            df = df[df[key].isin(selected_values)] 
            print('Keyword {}: using values {}'.format(key, selected_values))
    # create dict with selected values assumed by variables in opt['cfl']
    cfl = {}
    if opt['cfl'] is not None:
        for key in opt['cfl'].keys():
            print('Keyword {}: using '.format(key), end='')
            if opt['cfl'][key] == 'all':
                print('all values')
                selected_values = df[key].unique()
            else:
                selected_values = []
                for elem in opt['cfl'][key]:
                    print('{} '.format(elem), end='')
                    selected_values.append(elem)
            cfl[key] = selected_values
    # create a plot for every combination of values assumed by variables in opt['cfl'] (comparisons at file level)
    keys = list(cfl.keys())
    for values in list(itertools.product(*cfl.values())):
        dfc = df.copy()
        for i in range(len(keys)):
            dfc = dfc[dfc[keys[i]]==values[i]]
        plt.clf()
        if opt['hue'] is not None:
            palette = [colors[i] for i in range(dfc[plot_opt[0]].nunique())]
        else:
            palette=None
        ax = sns.lineplot(x='e', y=opt['plot'], hue=plot_opt[0], style=plot_opt[1], size=plot_opt[2], palette=palette, data=dfc)
        fm = ticker.ScalarFormatter()
        fm.set_scientific(False)
        ax.yaxis.set_major_formatter(fm)
        ax.yaxis.set_minor_formatter(fm)
        ax.set(xlabel='epoch', ylabel=opt['phase'] + ' ' + opt['plot'])
        #dir_ = os.path.join(opt['sd'], opt['exp'])
        dir_ = os.path.join(opt['sd'], 'loss_acc')
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        fn = get_filename_clean(keys, values, opt)
        plt.savefig(os.path.join(dir_, opt['plot'] + '_' + opt['phase'] + '__' + fn + '.pdf'), bbox_inches='tight', format='pdf')
        plt.close()

def compared_plot_train_val(opt):
    options = [opt[name] for name in ['hue','style','size'] if opt[name]]
    plot_opt = []
    for name in ['hue','style','size']:
        if opt[name]:
            plot_opt.append(list(opt[name].keys())[0])
        else:
            plot_opt.append(opt[name])
    # load data
    df = get_all_data_loss_top1(opt)
    # consider only specified values for opt['hue'], opt['size'], opt['size']
    for option in options:
        key = list(option.keys())[0]
        if option[key] == 'all':
            # use all values and do not filter
            print('Keyword {}: using all values'.format(key))
        else:
            # filter out non-selected values
            selected_values = []
            for elem in option[key]:
                selected_values.append(elem)
            df = df[df[key].isin(selected_values)] 
            print('Keyword {}: using values {}'.format(key, selected_values))
    # create dict with selected values assumed by variables in opt['cfl']
    cfl = {}
    if opt['cfl'] is not None:
        for key in opt['cfl'].keys():
            print('Keyword {}: using '.format(key), end='')
            if opt['cfl'][key] == 'all':
                print('all values')
                selected_values = df[key].unique()
            else:
                selected_values = []
                for elem in opt['cfl'][key]:
                    print('{} '.format(elem), end='')
                    selected_values.append(elem)
            cfl[key] = selected_values
    # create a plot for every combination of values assumed by variables in opt['cfl'] (comparisons at file level)
    keys = list(cfl.keys())
    for values in list(itertools.product(*cfl.values())):
        dfc = df.copy()
        for i in range(len(keys)):
            dfc = dfc[dfc[keys[i]]==values[i]]
        plt.clf()
        if opt['hue'] is not None:
            palette = [colors[i] for i in range(dfc[plot_opt[0]].nunique())]
        else:
            palette=None
        ax = sns.lineplot(x='e', y=opt['plot'], hue=plot_opt[0], style='phase', size=plot_opt[2], palette=palette, data=dfc)
        fm = ticker.ScalarFormatter()
        fm.set_scientific(False)
        ax.yaxis.set_major_formatter(fm)
        ax.yaxis.set_minor_formatter(fm)
        ax.set(xlabel='epoch', ylabel=opt['plot'])
        #dir_ = os.path.join(opt['sd'], opt['exp'])
        dir_ = os.path.join(opt['sd'], 'loss_acc')
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        fn = get_filename_clean(keys, values, opt)
        plt.savefig(os.path.join(dir_, opt['plot'] + '_train_val__' + fn + '.pdf'), bbox_inches='tight', format='pdf')
        plt.close()

def compared_class_count_hist(opt):
    # load data
    df, max_count, num_classes = get_all_data_counts(opt)
    # consider only specified values for opt['hue'], opt['size'], opt['size']
    if opt['hue'] is not None:
        key = list(opt['hue'].keys())[0]
        if opt['hue'][key] == 'all':
            # use all values and do not filter
            print('Keyword {}: using all values'.format(key))
        else:
            # filter out non-selected values
            selected_values = []
            for elem in opt['hue'][key]:
                selected_values.append(elem)
            df = df[df[key].isin(selected_values)] 
            print('Keyword {}: using values {}'.format(key, selected_values))
    # create dict with selected values assumed by variables in opt['cfl']
    cfl = {}
    if opt['cfl'] is not None:
        for key in opt['cfl'].keys():
            print('Keyword {}: using '.format(key), end='')
            if opt['cfl'][key] == 'all':
                print('all values')
                selected_values = df[key].unique()
            else:
                selected_values = []
                for elem in opt['cfl'][key]:
                    print('{} '.format(elem), end='')
                    selected_values.append(elem)
            cfl[key] = selected_values
    # create a plot for every combination of values assumed by variables in opt['cfl'] (comparisons at file level)
    keys = list(cfl.keys())
    for values in list(itertools.product(*cfl.values())):
        dfc = df.copy()
        for i in range(len(keys)):
            dfc = dfc[dfc[keys[i]]==values[i]]
        plt.clf()
        if opt['hue'] is not None:
            hue_values = dfc[list(opt['hue'].keys())[0]].unique()
            palette = [colors[i] for i in range(len(hue_values))]
        else:
            palette=None
        data = dfc.data.values
        epochs = opt['epochs']
        number_of_frames = len(epochs)

        def update_hist(i):
            plt.cla()
            idx = np.asarray([i for i in range(num_classes)])
            ax.set_xticks(idx)
            ax.set_xticklabels(idx)
            ax.set_xlabel('Class')
            ax.set_title('After {} epochs'.format(str(int(epochs[i])+1)))
            ax.set_ylim(top=max_count+0.05*max_count)
            ax.set_ylabel('Count')
            if opt['hue'] is not None:
                width = 0.2
                plots = []
                for j, hue_val in enumerate(hue_values):
                    dfc_current = dfc[dfc[list(opt['hue'].keys())[0]]==hue_val]
                    data = dfc_current.data.values
                    curr_plot = ax.bar(np.arange(0,num_classes)+j*width, data[i], color=palette[j], width=width)
                    plots.append(curr_plot[0])
                ax.legend(plots, hue_values, title=list(opt['hue'].keys())[0])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        anim = animation.FuncAnimation(fig, update_hist, number_of_frames, interval=1000, repeat=False)
        dir_ = os.path.join(opt['sd'], 'class_count')
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        fn = get_filename_clean(keys, values, opt)
        anim.save(os.path.join(dir_, opt['plot'] + '__' + fn + '.gif'), writer='imagemagick')



