from IPython import embed
import argparse
import os

parser = argparse.ArgumentParser(description='count_visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp', default=f'..{os.sep}results', required=True)
opt = vars(parser.parse_args())

os.system(f'python3 count_visualization.py --exp ' + opt['exp'])
#os.system(f'python3 print_outliers.py --exp ' + opt['exp'])
if 'corr' in opt['exp']:
    os.system(f'python3 check_masks.py --exp ' + opt['exp'])

os.system(f'python3 organize_experiment_analysis.py --exp ' + opt['exp'])


