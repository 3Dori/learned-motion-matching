from networks.base_trainer import BaseTrainer
from networks.learned_motion_AE import Projector
from networks.utils import all_sequences_of_dataset
import common.locomotion_utils as utils
from common.locomotion_utils import nearest_frame

import numpy as np
import torch

import sys


class ProjectorTrainer(BaseTrainer):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size

    @staticmethod
    def prepare_batches(dataset, sequences, batch_size):
        buffer_x = np.zeros((batch_size, utils.X_LEN))

        probs = []
        for i, (subject, action) in enumerate(sequences):
            probs.append(dataset[subject][action]['n_frames'])
        probs = np.array(probs) / np.sum(probs)

        while True:
            idxs = np.random.choice(len(sequences), size=batch_size, replace=True, p=probs)
            # randomly pick batch_size pairs of frames

            for i, (subject, action) in enumerate(sequences[idxs]):
                action = dataset[subject][action]
                frame = np.random.randint(0, action['n_frames'] - 1)
                # add noise
                n_sigma = np.random.uniform()
                n = np.random.normal(size=utils.X_LEN)
                buffer_x[i] = action['input_feature'][frame] + n_sigma * n
            yield buffer_x

    def train(
            self, dataset, batch_size=32, epochs=100000, lr=0.001, seed=0,
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
        fps = dataset.fps()

        sequences = all_sequences_of_dataset(dataset)
        batches = self.prepare_batches(dataset, sequences, batch_size)

        projector_mean_in = torch.tensor(dataset.projector_mean_in, dtype=torch.float32).to(device)
        projector_std_in = torch.tensor(dataset.projector_std_in, dtype=torch.float32).to(device)
        projector_mean_out = torch.tensor(dataset.projector_mean_out, dtype=torch.float32).to(device)
        projector_std_out = torch.tensor(dataset.projector_std_out, dtype=torch.float32).to(device)

        rolling_loss = None
        for epoch in range(epochs):
            optimizer.zero_grad()

            x_in = next(batches)
            nearest_x, nearest_z = nearest_frame(dataset, x_in)
            x_in = torch.tensor(x_in).to(device)
            nearest_x = torch.tensor(nearest_x, dtype=torch.float32).to(device)
            nearest_z = torch.tensor(nearest_z, dtype=torch.float32).to(device)

            x_z_out = projector(x_in)
            x_out = x_z_out[:, :utils.X_LEN]
            z_out = x_z_out[:, utils.X_LEN:]

            loss_xval = torch.mean(torch.abs(nearest_x - x_out))
            loss_zval = torch.mean(torch.abs(nearest_z - z_out))
            loss_dist = torch.mean(torch.abs(torch.square(x_in - nearest_x) - torch.square(x_in - x_out)))

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
