from IPython import embed
import argparse
import os

parser = argparse.ArgumentParser(description='gradients_stats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp', default=f'..{os.sep}results', required=True)
parser.add_argument('--bs', default=128, type=int, required=True)
parser.add_argument('--gpu', default=0, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
opt = vars(parser.parse_args())

print(f"You can choose the batch size for the gradients histograms!!! Now you have BS = {opt['bs']}")

if 'cifar10' in opt['exp']:
    dataset = 'cifar10'
elif 'cifar100' in opt['exp']:
    dataset = 'cifar100'
else:
    raise NotImplementedError('Gradients loader are only for cifar10/100')

if 'resnet18' in opt['exp']:
    arch = 'resnet18'
elif 'resnet34' in opt['exp']:
    arch = 'resnet34'
else:
    raise  NotImplementedError('Gradients loader have only been tested for resnet')

experiment = opt['exp'].split(f'_{arch}')[0]

command = f"python3 computing_gradients_stats.py with dataset.name='{dataset}' arch='{arch}' exp={experiment} g={opt['gpu']}"

exp_path = os.path.join('..', 'results', opt['exp'])
for run in os.listdir(exp_path):
    if not 'analysis' in run:
        for file in os.listdir(os.path.join(exp_path, run)):
            if 'model_' in file:
                arguments = f" run='{run}' resume='{file}' b={opt['bs']}"
                print(f"Processing model: {file}")
                os.system(command + arguments)
