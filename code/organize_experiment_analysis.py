import argparse
import os
from IPython import embed
import glob2
from shutil import copyfile


def get_paths_to_be_saved(run):
    files_to_be_saved = []
    files_to_be_saved += glob2.glob(f'{run}/**/*.log')
    files_to_be_saved += glob2.glob(f'{run}/**/*.pdf')
    return files_to_be_saved

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--input_dir', type=str, required=True) # Choose the experiment
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    exp_path =  os.path.join(f'..{os.sep}results{os.sep}', args.input_dir)

    if not args.output_dir:
        save_path =os.path.join(exp_path, 'analysis_experiments')

    # Get all run paths you need to get logs and pdf (general things to be saved)
    for i, run in enumerate(os.listdir(exp_path)):
        save_run_path = os.path.join(save_path, str(i))
        os.makedirs(save_run_path, exist_ok=True)
        if not run == 'analysis_experiments':
            for file in get_paths_to_be_saved(os.path.join(exp_path, run)):
                copyfile(file, os.path.join(save_run_path, file.split(f'{os.sep}')[-1]))

