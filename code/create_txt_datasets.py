import os, json, argparse
from IPython import embed
from constants import CLASS_DICT

for dataset in CLASS_DICT:
    CLASS_DICT[dataset] = dict(zip(list(CLASS_DICT[dataset].values()), CLASS_DICT[dataset].keys()))

'''
Create a .txt with images' path
'''

datasets = {'CINIC': '/home/matteo/data/CINIC'}

parser = argparse.ArgumentParser(description='dataset_to_txt',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='', required=True)
opt = vars(parser.parse_args())


folders = ['train','test','val']
data = opt['dataset']
for fol in folders:
    f = open(f'data/{data}_{fol}.txt','w+')
    tmp = os.path.join(datasets[opt['dataset']],fol)
    for idx, cl in enumerate(os.listdir(tmp)):
        for img_name in os.listdir(os.path.join(tmp,cl)):
            cl_idx = CLASS_DICT[data][cl]
            line = f'{fol}/{cl}/{img_name} {cl_idx}\n'
            f.write(line)

