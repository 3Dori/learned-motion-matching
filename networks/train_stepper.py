import sys

from networks.learned_motion_AE import Stepper
from networks.utils import randomly_pick_from_dataset, all_sequences_of_dataset
import common.locomotion_utils as utils

import numpy as np
import torch


class StepperTrainer(object):
    def __init__(self, compressor, dataset, hidden_size=512):
        self.compressor = compressor
        self.hidden_size = hidden_size

        # generate z features
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            for subject in dataset.subjects():
                for action in subject.values():
                    y = torch.tensor(action['Y_feature'][np.newaxis]).to(device)
                    q = torch.tensor(action['Q_feature'][np.newaxis]).to(device)

                    z = compressor(torch.cat([y, q], dim=-1))
                    action['Z_code'] = z.reshape((-1, utils.Z_LEN))

    @staticmethod
    def prepare_batches(dataset, sequences, batch_size, window):
        buffer_x_z = np.zeros((batch_size, window, utils.X_LEN + utils.Z_LEN), dtype=np.float32)

        probs = []
        for i, (subject, action) in enumerate(sequences):
            probs.append(dataset[subject][action]['n_frames'])
        probs = np.array(probs) / np.sum(probs)

        while True:
            idxs = np.random.choice(len(sequences), size=batch_size, replace=True, p=probs)
            # randomly pick batch_size pairs of frames
            for i, (subject, action) in enumerate(sequences[idxs]):
                action = dataset[subject][action]
                frame = np.random.randint(0, action['n_frames'] - window)
                buffer_x_z[i, :, :utils.X_LEN] = action['input_feature'][frame:frame+window]
                buffer_x_z[i, :, utils.X_LEN:] = action['Z_code'][frame:frame+window]
            yield buffer_x_z

    def train(
            self, dataset, batch_size=32, window=2, epochs=10000, lr=0.001, seed=0,
            w_xval=1.0,
            w_zval=1.0,
            w_xvel=1.0,
            w_zvel=1.0
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)

        stepper = Stepper(param_size=utils.X_LEN + utils.Z_LEN, hidden_size=self.hidden_size)
        optimizer = torch.optim.AdamW(
            stepper.parameters(),
            lr=lr,
            amsgrad=True,
            weight_decay=0.001
        )

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        fps = dataset.fps()

        sequences = all_sequences_of_dataset(dataset)
        batches = self.prepare_batches(dataset, sequences, window, batch_size)

        for epoch in range(epochs):
            optimizer.zero_grad()

            x_z = next(batches)
            x_z = torch.tensor(x_z).to(device)
            predicted_x_z = [x_z[:, 0:1]]
            for i in range(window - 1):
                delta_x_z = stepper(predicted_x_z[-1])
                predicted_x_z.append(predicted_x_z[-1] + delta_x_z)
            predicted_x_z = torch.cat(predicted_x_z, dim=1)
            x = x_z[:, :, :utils.X_LEN]
            z = x_z[:, :, utils.X_LEN:]
            predicted_x = predicted_x_z[:, :, :utils.X_LEN]
            predicted_z = predicted_x_z[:, :, utils.X_LEN:]

            def velocity(v):
                return (v[:, :-1] - v[:, 1:]) * fps
            x_vel = velocity(x)
            z_vel = velocity(z)
            predicted_x_vel = velocity(predicted_x)
            predicted_z_vel = velocity(predicted_z)

            loss_xval = torch.mean(torch.abs(x - predicted_x))
            loss_zval = torch.mean(torch.abs(z - predicted_z))
            loss_xvel = torch.mean(torch.abs(x_vel - predicted_x_vel))
            loss_zvel = torch.mean(torch.abs(z_vel - predicted_z_vel))

            loss = (
                loss_xval * w_xval +
                loss_zval * w_zval +
                loss_xvel * w_xvel +
                loss_zvel * w_zvel
            )
            loss.backward()
            optimizer.step()

            # logging
            if epoch % 10 == 0:
                sys.stdout.write('\rIter: %7i Loss: %5.3f' % (epoch, loss.item()))

            if epoch % 1000 == 0:
                scheduler.step()

        return stepper