import glob2, os, subprocess, argparse
from IPython import embed

parser = argparse.ArgumentParser(description='exp extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dropbox', default=False)
opt = vars(parser.parse_args())

path = '../results'
folders = glob2.glob('../results/CVPR*')

for exp_path in folders:
    #exp_path = os.path.join(path,exp)
    exp = exp_path.split('/')[-1]
    os.system(f"python organize_experiment_analysis.py --exp {exp}")

    if opt['dropbox']:
        os.system(f"cp -r {exp_path} ~/Dropbox/tmp_results/{exp}")

