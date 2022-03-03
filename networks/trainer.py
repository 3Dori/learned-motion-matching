from networks.learned_motion_AE import Compressor, Decompressor
import numpy as np
import torch


class CompressorTrainer(object):
    def __init__(self, skeleton):
        self.skeleton = skeleton
        n_joints = self.skeleton.num_joints()
        self.n_features = 6 + 6 + 6 + 6 + 3
        self.n_outputs = n_joints * (1)

    def prepare_batches(self, batch_size, dataset, target_length, sequences):
        buffer_x = np.zeros((batch_size, target_length, self.n_features))
        buffer_y = np.zeros((batch_size, target_length, self.n_outputs))

        probs = []
        for i, (subject, action) in enumerate(sequences):
            probs.append(dataset[subject][action]['rotations'].shape[0])
        probs = np.array(probs) / np.sum(probs)

