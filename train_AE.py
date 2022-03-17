import torch.cuda

from long_term.dataset_locomotion import dataset, actions_valid
from long_term.locomotion_utils import build_extra_features, compute_input_features, compute_output_features


if __name__ == '__main__':
    prefix_length = 30
    target_length = 60
    future_trajectory_length = 60

    if torch.cuda.is_available():
        dataset.cuda()

    sequences_train = []
    sequences_valid = []
    n_discarded = 0
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            if dataset[subject][action]['rotations'].shape[0] < prefix_length + target_length + future_trajectory_length:
                n_discarded += 1
                continue

            train = True
            for action_valid in actions_valid:
                if action.startswith(action_valid):
                    train = False
                    break
            if train:
                sequences_train.append((subject, action))
            else:
                sequences_valid.append((subject, action))

    print('%d sequences were discarded for being too short.' % n_discarded)
    print('Training on %d sequences, validating on %d sequences.' % (len(sequences_train), len(sequences_valid)))
    dataset.compute_positions()
    dataset.compute_velocities()
    build_extra_features(dataset)
    compute_input_features(dataset)
    compute_output_features(dataset)
    print('Done')
