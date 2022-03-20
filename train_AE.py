import torch.cuda

from datasets.dataset_locomotion import dataset, actions_valid
from datasets.locomotion_utils import build_extra_features, compute_input_features, compute_output_features

import os.path
import pickle

from networks.train_decompressor import DecompressorTrainer


def load_dataset():
    if os.path.exists('datasets/dataset_learned_motion.pkl'):
        with open('datasets/dataset_learned_motion.pkl', 'rb') as f:
            return pickle.load(f)
    else:
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
        print('Computing output features')
        compute_output_features(dataset)
        with open('datasets/dataset_learned_motion.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        return dataset


if __name__ == '__main__':
    dataset = load_dataset()
    decompressor_trainer = DecompressorTrainer()
    print('Start training')
    decompressor_trainer.train(dataset)
    print('Done')
