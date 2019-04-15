import numpy as np
import json, logging, os, subprocess
import time

def gitrev(opt):
    cmds = [['git', 'rev-parse', 'HEAD'],
            ['git', 'status'],
            ['git', 'diff']]
    rs = []
    for c in cmds:
        subp = subprocess.Popen(c,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
        r, _ = subp.communicate()
        rs.append(r)

    rs[0] = rs[0].strip()
    return rs

def create_basic_logger(ctx, filename, idx=0):
    opt = ctx.opt
    #d = os.path.join(opt.get('o'), opt['arch']) +'/logs'
    d = os.path.join(opt.get('o'), opt['filename']) +'/logs'
    if not os.path.isdir(d):
        os.makedirs(d)

    #fn = os.path.join(d, filename + '.log')
    fn = os.path.join(d, 'flogger.log')
    l = logging.getLogger('%s'%idx)
    l.propagate = False

    fh = logging.FileHandler(fn)
    fmt = logging.Formatter('%(message)s')
    fh.setFormatter(fmt)
    l.setLevel(logging.INFO)
    l.addHandler(fh)
    ctx.ex.info['weights_logger'] = fn
    return l



def create_logger(ctx, idx=0):
    opt = ctx.opt
    if not opt.get('fl', None):
        return

    if len(opt.get('resume', '')) > 0:
        print('Retraining, will stop logging')
        return

    if opt.get('filename', None) is None:
        build_filename(ctx)

    #d = os.path.join(opt.get('o'), opt['arch']) +'/logs'
    d = os.path.join(opt.get('o'), opt['filename']) +'/logs'
    if not os.path.isdir(d):
        os.makedirs(d)
    #fn = os.path.join(d, opt['filename']+'.log')
    fn = os.path.join(d, 'flogger.log')
    l = logging.getLogger('%s'%idx)
    l.propagate = False

    fh = logging.FileHandler(fn)
    fmt = logging.Formatter('%(message)s')
    fh.setFormatter(fmt)
    l.setLevel(logging.INFO)
    l.addHandler(fh)

    r = gitrev(opt)
    l.info('SHA %s'%r[0])
    l.info('STATUS %s'%r[1])
    l.info('DIFF %s'%r[2])

    l.info('')
    l.info('[OPT] ' + json.dumps(opt))
    l.info('')
    ctx.ex.logger = l
    return l


#def schedule(opt, e, logger=None, k=None):
def schedule(ctx, k=None):
    logger = ctx.ex.logger
    e = ctx.epoch
    opt = ctx.opt
    ks = k + 's'
    if opt[ks] == '':
        opt[ks] = json.dumps([[opt['epochs'], opt[k]]])

    if isinstance(opt[ks], str):
        rs = json.loads(opt[ks])
    else:
        rs = opt[ks]

    idx = len(rs)-1
    for i in range(1,len(rs)):
        if e < rs[i][0]:
            idx = i-1
            break
    if e >= rs[len(rs)-1][0]:
        idx = i

    r = rs[idx][1]
    return r


def init_opt(ctx):
    cfg = ctx.ex.current_run.config
    opt = dict()
    for k,v in cfg.items():
        opt[k] = v
    return opt

def build_filename(ctx):
    opt = ctx.opt
    whitelist = opt['whitelist']
    marker = opt['marker']
    dconf = dict()
    cfg_mdf = ctx.ex.current_run.config_modifications.modified
    # todo: fix this for sacred
    for k in cfg_mdf:
        if 'dataset' in k:
            if opt['dataset']['name'] is not 'cifar10':
                dconf['dataset'] = opt['dataset']['name']
            if opt['dataset']['perc'] > 0:
                dconf['perc'] = opt['dataset']['perc']
            if opt['dataset']['norm'] > 0:
                dconf['norm'] = opt['dataset']['norm']
        else:
            dconf[k] = opt[k]

    base_whilelist = ['dataset', 'arch']
    blacklist = ['g','save', 'fl', 'tfl', 'dbl', 'o', 'source', '__doc__', 'j', 'print_freq']
    
    dconfc = dconf.copy()
    for k in dconf.keys():
        # if k in blacklist:
        #     oc.pop(k, None)
        if k in blacklist:
            dconfc.pop(k, None)
    dconf = dconfc
    from ast import literal_eval
    whitelist = literal_eval(whitelist)
    whitelist += base_whilelist 
    
    o = json.loads(json.dumps(opt))
    oc = o.copy()
    for k in o.keys():
        # if k in blacklist:
        #     oc.pop(k, None)
        if k not in whitelist:
            oc.pop(k, None)

        if k == 'dataset':
            oc[k] = oc[k]['name']
    o = oc
    o = {**o, **dconf}
    

    t = ''
    if not marker == '':
        t = marker + '_'
    t = t + time.strftime('(%b_%d_%H_%M_%S)') + '_opt_'
    opt['time'] = t
    opt['filename'] = t + json.dumps(o, sort_keys=True,
                separators=(',', ':'))
    print(opt['filename'])
