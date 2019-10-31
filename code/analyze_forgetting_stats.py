import argparse
import pickle as pkl
import os
import numpy as np

from IPython import embed


def compute_forgetting_statistics(stats, npresentations):
    presentations_needed_to_learn = {}
    unlearned_per_presentation = {}
    margins_per_presentation = {}
    first_learned = {}
    for example_id, example_stats in stats.items():
        # Forgetting event is a transition in accuracy from 1 to 0
        presentation_acc = np.array(example_stats['acc'][:npresentations])
        transitions = presentation_acc[1:] - presentation_acc[:-1]

        embed()

        # Find all presentations when forgetting occurs
        if len(np.where(transitions == -1)[0]) > 0:
            unlearned_per_presentation[example_id] = np.where(
                transitions == -1)[0] + 2
        else:
            unlearned_per_presentation[example_id] = []

        # Find number of presentations needed to learn example,
        # e.g. last presentation when acc is 0
        if len(np.where(presentation_acc == 0)[0]) > 0:
            presentations_needed_to_learn[example_id] = np.where(
                presentation_acc == 0)[0][-1] + 1
        else:
            presentations_needed_to_learn[example_id] = 0

        # Find the misclassication margin for each presentation of the example
        margins_per_presentation[example_id] = np.array(example_stats['margin'][:npresentations])

        # Find the presentation at which the example was first learned,
        # e.g. first presentation when acc is 1
        if len(np.where(presentation_acc == 1)[0]) > 0:
            first_learned[example_id] = np.where(
                presentation_acc == 1)[0][0]
        else:
            first_learned[example_id] = np.nan

    return presentations_needed_to_learn, unlearned_per_presentation, margins_per_presentation, first_learned

# Sorts examples by number of forgetting counts during training, in ascending order
# If an example was never learned, it is assigned the maximum number of forgetting counts
# If multiple training runs used, sort examples by the sum of their forgetting counts over all runs
#
# unlearned_per_presentation_all: list of dictionaries, one per training run
# first_learned_all: list of dictionaries, one per training run
# npresentations: number of training epochs
#
# Returns 2 numpy arrays containing the sorted example ids and corresponding forgetting counts
#
def sort_examples_by_forgetting(unlearned_per_presentation_all,
                                first_learned_all, npresentations):

    # embed()

    # Initialize lists
    example_original_order = []
    example_stats = []

    for example_id in unlearned_per_presentation_all.keys():

        # Add current example to lists
        example_original_order.append(example_id)
        example_stats.append(0)

        # Get all presentations when current example was forgotten during current training run
        stats = unlearned_per_presentation_all[example_id]

        # If example was never learned during current training run, add max forgetting counts
        if np.isnan(first_learned_all[example_id]):
            example_stats[-1] += npresentations
        else:
            example_stats[-1] += len(stats)

    print('Number of unforgettable examples: {}'.format(
        len(np.where(np.array(example_stats) == 0)[0])))
    return np.array(example_original_order)[np.argsort(
        example_stats)], np.sort(example_stats)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options")
    # parser.add_argument('--input_dir', type=str, required=True)
    # parser.add_argument('--output_dir', type=str, required=True)
    # parser.add_argument(
    #     '--output_name',
    #     type=str,
    #     required=True)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()


    exp = 'CVPR_sampler_resnet18_cifar100'
    run = '(ott_30_14_21_24)_opt_{"arch":"resnet18","dataset":"cifar100","exp":"CVPR_sampler","normalizer":true,"sampler":"invtunnel","temperature":0.1}'
    path = os.path.join(f'..{os.sep}results', exp, run, 'forgetting_stats')

    with open(os.path.join(path, 'stats.pkl'), 'rb') as fin:
        loaded = pkl.load(fin)

    embed()

    # Compute the forgetting statistics per example for training run
    _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(
        loaded, args.epochs)

    # embed()

    # Sort examples by forgetting counts in ascending order, over one or more training runs
    ordered_examples, ordered_values = sort_examples_by_forgetting(
        unlearned_per_presentation, first_learned, args.epochs)


    # embed()
