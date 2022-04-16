from common.dataset_locomotion import get_valid_sequences
from networks.base_trainer import BaseTrainer
from networks.learned_motion_AE import Projector
import common.locomotion_utils as utils

import numpy as np
import torch
from sklearn.neighbors import BallTree

import sys

from networks.utils import PROJECTOR_PATH


class ProjectorTrainer(BaseTrainer):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size

    @staticmethod
    def get_valid_x_z_from_dataset(dataset, sequences):
        n_total_frames = sum(dataset[subject][action]['n_frames'] for subject, action in sequences)
        X = np.zeros((n_total_frames, utils.X_LEN), dtype=np.float32)
        Z = np.zeros((n_total_frames, utils.Z_LEN), dtype=np.float32)
        offset = 0

        for subject, action in sequences:
            action = dataset[subject][action]
            X[offset:offset+action['n_frames']] = action['input_feature']
            Z[offset:offset+action['n_frames']] = action['Z_code']
            offset += action['n_frames']
        # or we can do this:
        # X = [action['input_feature'] for subject in dataset.subjects() for action in dataset[subject].values()]
        # X = np.concatenate(X, axis=0)
        return X, Z

    @staticmethod
    def prepare_batches(X, batch_size):
        n_total_frames = X.shape[0]
        buffer_x = np.zeros((batch_size, utils.X_LEN), dtype=np.float32)

        while True:
            idxs = np.random.choice(n_total_frames, size=batch_size)
            n_sigma = np.random.uniform(size=(batch_size, 1))
            n = np.random.normal(size=(batch_size, utils.X_LEN))
            buffer_x[:] = X[idxs] + n_sigma * n
            yield buffer_x

    def train(
            self, dataset, projector=None, batch_size=32, epochs=10000, lr=0.001, seed=0,
            w_xval=1.0,
            w_zval=5.0,
            w_dist=0.3
        ):
        np.random.seed(seed)
        torch.manual_seed(seed)

        if projector is None:
            projector = Projector(dataset,
                                  input_size=utils.X_LEN,
                                  output_size=utils.X_LEN + utils.Z_LEN,
                                  hidden_size=self.hidden_size)
        optimizer = torch.optim.AdamW(
            projector.parameters(),
            lr=lr,
            amsgrad=True,
            weight_decay=0.001
        )

        device = dataset.device()
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        sequences, _ = get_valid_sequences(dataset)
        X, Z = self.get_valid_x_z_from_dataset(dataset, sequences)
        batches = self.prepare_batches(X, batch_size)
        search_tree = BallTree(X)

        rolling_loss = None
        for epoch in range(epochs):
            optimizer.zero_grad()

            x_hat = next(batches)
            nearest_idx = search_tree.query(x_hat, k=1, return_distance=False)[:, 0]
            nearest_x = torch.as_tensor(X[nearest_idx], dtype=torch.float32, device=device)
            nearest_z = torch.as_tensor(Z[nearest_idx], dtype=torch.float32, device=device)
            x_hat = torch.as_tensor(x_hat, dtype=torch.float32, device=device)

            x_z_out = projector.project(x_hat)
            x_out = x_z_out[:, :utils.X_LEN]
            z_out = x_z_out[:, utils.X_LEN:]

            loss_xval = torch.mean(torch.abs(nearest_x - x_out))
            loss_zval = torch.mean(torch.abs(nearest_z - z_out))
            loss_dist = torch.mean(torch.abs(torch.sum(torch.square(x_hat - nearest_x), dim=-1) -
                                             torch.sum(torch.square(x_hat - x_out), dim=-1)))

            loss = (
                loss_xval * w_xval +
                loss_zval * w_zval +
                loss_dist * w_dist
            )
            loss.backward()
            optimizer.step()

            if rolling_loss is None:
                rolling_loss = loss.item()
            else:
                rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01

            if epoch % 1000 == 0:
                scheduler.step()

            # logging
            if epoch % 10 == 0:
                sys.stdout.write('\rIter: %7i Loss: %5.3f' % (epoch, rolling_loss))
            if epoch % 1000 == 0:
                torch.save(projector, PROJECTOR_PATH)

        sys.stdout.write('\n')
        return projector
