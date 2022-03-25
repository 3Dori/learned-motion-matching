from networks.base_trainer import BaseTrainer
from networks.learned_motion_AE import Projector
from networks.utils import all_sequences_of_dataset
import common.locomotion_utils as utils
from common.locomotion_utils import nearest_frame

import numpy as np
import torch
from sklearn.neighbors import BallTree

import sys


class ProjectorTrainer(BaseTrainer):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size

    @staticmethod
    def get_all_x_z_from_dataset(dataset):
        X = np.zeros((dataset.n_total_frames, utils.X_LEN), dtype=np.float32)
        Z = np.zeros((dataset.n_total_frames, utils.Z_LEN), dtype=np.float32)
        offset = 0
        for subject in dataset.subjects():
            for action in dataset[subject].values():
                X[offset:offset+action['n_frames']] = action['input_feature']
                Z[offset:offset+action['n_frames']] = action['Z_code']
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
            self, dataset, batch_size=32, epochs=10000, lr=0.001, seed=0,
            w_xval=1.0,
            w_zval=5.0,
            w_dist=0.3
        ):
        np.random.seed(seed)
        torch.manual_seed(seed)

        projector = Projector(input_size=utils.X_LEN,
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

        X, Z = self.get_all_x_z_from_dataset(dataset)
        batches = self.prepare_batches(X, batch_size)
        search_tree = BallTree(X)

        projector_mean_in = torch.as_tensor(dataset.projector_mean_in, dtype=torch.float32, device=device)
        projector_std_in = torch.as_tensor(dataset.projector_std_in, dtype=torch.float32, device=device)
        projector_mean_out = torch.as_tensor(dataset.projector_mean_out, dtype=torch.float32, device=device)
        projector_std_out = torch.as_tensor(dataset.projector_std_out, dtype=torch.float32, device=device)

        rolling_loss = None
        for epoch in range(epochs):
            optimizer.zero_grad()

            x_hat = next(batches)
            nearest_idx = search_tree.query(x_hat, k=1, return_distance=False)[:, 0]
            nearest_x = torch.as_tensor(X[nearest_idx], dtype=torch.float32, device=device)
            nearest_z = torch.as_tensor(Z[nearest_idx], dtype=torch.float32, device=device)
            x_hat = torch.as_tensor(x_hat, dtype=torch.float32, device=device)

            x_z_out = (projector((x_hat - projector_mean_in) / projector_std_in)
                       * projector_std_out + projector_mean_out)
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

        sys.stdout.write('\n')
        return projector
