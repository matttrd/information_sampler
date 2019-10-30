import os, sys, subprocess, json, argparse, time
from itertools import product
import torch as th
from multiprocessing.pool import ThreadPool

parser = argparse.ArgumentParser(description='MultipleJobber',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i','--input',   help='processes to run', type=str, required=True)
parser.add_argument('--gpus',         help='array of gpus to use', type=str, default='')
parser.add_argument('-j', '--max_jobs',     help='max jobs',    type=int, default = 1)
parser.add_argument('-r', '--run',    help='run',  action='store_true')
opt = vars(parser.parse_args())


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def run_cmds(cmds, max_cmds):
    for cs in list(chunks(cmds, max_cmds)):
        ps = []
        try:
            for c in cs:
                p = subprocess.Popen(c, shell=True)
                ps.append(p)

            for p in ps:
                p.wait()

        except KeyboardInterrupt:
            print('Killing everything')
            for p in ps:
                p.kill()
            sys.exit()

if opt['gpus'] == '':
    gs = range(th.cuda.device_count())
else:
    gs = json.loads(opt['gpus'])

with open(opt['input'], 'r') as f:
    cmds = f.readlines()

cmds = [x.strip() for x in cmds if x[0] != '#']
cmdss = []

for c in cmds:
    c = c + (' g=%d')%(gs[len(cmdss)%len(gs)])
    cmdss.append(c)

if not opt['run']:
    for c in cmdss:
        print(c)
else:
    run_cmds(cmdss, opt['max_jobs'])