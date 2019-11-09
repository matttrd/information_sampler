from IPython import embed
import argparse
import os

#Check if gradient stats have been created
def is_folder_present(checking_fodler):
    found = False
    exp_path = os.path.join('..', 'results', opt['exp'])
    for run in os.listdir(exp_path):
        for folder in os.listdir(os.path.join(exp_path, run)):
            if checking_fodler in folder:
                found = True
    return found


parser = argparse.ArgumentParser(description='count_visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp', default=f'..{os.sep}results', required=True)
parser.add_argument('--replace', default=False, required=True)
opt = vars(parser.parse_args())

# Plots that are performed regardless your decision
os.system(f'python3 count_visualization.py --exp ' + opt['exp'])
if 'corr' in opt['exp']:
    os.system(f'python3 check_masks.py --exp ' + opt['exp'])

# Plots performed only if necessary or if required
if not is_folder_present('analysis_exctracted_images') or bool(opt['replace']):
    os.system(f'python3 print_outliers.py --exp ' + opt['exp'])

if not is_folder_present('gradients_stats') or bool(opt['replace']):
    os.system(f'python3 computing_gradients_stats_experiment.py --exp ' + opt['exp'])

os.system(f'python3 print_gradients.py --exp ' + opt['exp'])

#Plotting train and validaiton losses
global_path = os.getcwd()
global_path = global_path[:-4] + 'results'
global_path = os.path.join(global_path, opt['exp'])

os.system(f'python3 single_data_viz.py  --exp_dir  {global_path} --plot loss')
os.system(f'python3 single_data_viz.py  --exp_dir  {global_path} --plot top1')


os.system(f'python3 organize_experiment_analysis.py --exp ' + opt['exp'])
