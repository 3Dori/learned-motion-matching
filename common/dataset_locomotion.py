# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle

import torch
import numpy as np

from common.locomotion_utils import build_extra_features, compute_input_features, compute_output_features
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset

# Set to True for validation.
# There is no test set here, since we do not evaluate numerically
# for long-term generation of locomotion.
perform_validation = False

if perform_validation:
    actions_valid = ['jog_1', 'walk_4', 'run_1']
else:
    actions_valid = []

# Note: the "joints_left" and "joint_right" indices refer to the optimized skeleton
# after calling "remove_joints".
skeleton_cmu = Skeleton(offsets=[
        [ 0. , 0. , 0. ],
        [ 0.      ,  0.      ,  0.      ],
        [ 1.36306 , -1.79463 ,  0.83929 ],
        [ 2.44811 , -6.72613 ,  0.      ],
        [ 2.5622  , -7.03959 ,  0.      ],
        [ 0.15764 , -0.43311 ,  2.32255 ],
        [ 0.      ,  0.      ,  0.      ],
        [-1.30552 , -1.79463 ,  0.83929 ],
        [-2.54253 , -6.98555 ,  0.      ],
        [-2.56826 , -7.05623 ,  0.      ],
        [-0.16473 , -0.45259 ,  2.36315 ],
        [ 0.      ,  0.      ,  0.      ],
        [ 0.02827 ,  2.03559 , -0.19338 ],
        [ 0.05672 ,  2.04885 , -0.04275 ],
        [ 0.      ,  0.      ,  0.      ],
        [-0.05417 ,  1.74624 ,  0.17202 ],
        [ 0.10407 ,  1.76136 , -0.12397 ],
        [ 0.      ,  0.      ,  0.      ],
        [ 3.36241 ,  1.20089 , -0.31121 ],
        [ 4.983   , -0.      , -0.      ],
        [ 3.48356 , -0.      , -0.      ],
        [ 0.      ,  0.      ,  0.      ],
        [ 0.71526 , -0.      , -0.      ],
        [ 0.      ,  0.      ,  0.      ],
        [ 0.      ,  0.      ,  0.      ],
        [-3.1366  ,  1.37405 , -0.40465 ],
        [-5.2419  , -0.      , -0.      ],
        [-3.44417 , -0.      , -0.      ],
        [ 0.      ,  0.      ,  0.      ],
        [-0.62253 , -0.      , -0.      ],
        [ 0.      ,  0.      ,  0.      ],
    ],
    parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 15,
       13, 17, 18, 19, 20, 21, 20, 13, 24, 25, 26, 27, 28, 27],
    joints_left = [6, 7, 8, 9, 10, 21, 22, 23, 24, 25],
    joints_right = [1, 2, 3, 4, 5, 16, 17, 18, 19, 20]
    )

dataset_path = 'datasets/dataset_locomotion.npz'
long_term_weights_path = 'weights_long_term.bin'
dataset = MocapDataset(dataset_path, skeleton_cmu, fps=120)

# Remove useless joints, from both the skeleton and the dataset
skeleton_cmu.remove_joints([13, 21, 23, 28, 30], dataset)

dataset.mirror()
dataset.compute_euler_angles('yzx')
dataset.downsample(4)


def load_dataset():
    if os.path.exists('datasets/dataset_learned_motion.pkl'):
        with open('datasets/dataset_learned_motion.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        if torch.cuda.is_available():
            dataset.cuda()
        dataset.compute_positions()
        dataset.compute_velocities()
        build_extra_features(dataset)
        compute_input_features(dataset)
        print('Computing output features')
        compute_output_features(dataset)
        with open('datasets/dataset_learned_motion.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        return dataset


def get_valid_sequences(dataset):
    prefix_length = 30
    target_length = 60
    future_trajectory_length = 60

    sequences_train = []
    sequences_valid = []
    n_discarded = 0
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            if dataset[subject][action]['n_frames'] < prefix_length + target_length + future_trajectory_length:
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
    return np.array(sequences_train), np.array(sequences_valid)
