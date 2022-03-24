import sys

from networks.learned_motion_AE import Stepper
from networks.utils import all_sequences_of_dataset
import common.locomotion_utils as utils

import numpy as np
import torch


class StepperTrainer(object):
    def __init__(self, compressor, hidden_size=512):
        self.compressor = compressor
        self.hidden_size = hidden_size

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
            self, dataset, batch_size=32, window=10, epochs=100000, lr=0.001, seed=0,
            w_xval=2.0,
            w_zval=7.5,
            w_xvel=0.2,
            w_zvel=0.5
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

        device = dataset.device()
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        fps = dataset.fps()

        sequences = all_sequences_of_dataset(dataset)
        batches = self.prepare_batches(dataset, sequences, batch_size, window)

        stepper_mean_in = torch.tensor(dataset.stepper_mean_in, dtype=torch.float32).to(device)
        stepper_std_in = torch.tensor(dataset.stepper_std_in, dtype=torch.float32).to(device)
        stepper_mean_out = torch.tensor(dataset.stepper_mean_out, dtype=torch.float32).to(device)
        stepper_std_out = torch.tensor(dataset.stepper_std_out, dtype=torch.float32).to(device)

        rolling_loss = None
        for epoch in range(epochs):
            optimizer.zero_grad()

            x_z = next(batches)    # batch_size * window * (len(x) + len(z))
            x_z = torch.tensor(x_z).to(device)
            predicted_x_z = [x_z[:, 0:1]]
            for i in range(window - 1):
                delta_x_z = (stepper((predicted_x_z[-1] - stepper_mean_in) / stepper_std_in)
                             * stepper_std_out + stepper_mean_out)
                predicted_x_z.append(predicted_x_z[-1] + delta_x_z / fps)
            predicted_x_z = torch.cat(predicted_x_z, dim=1)
            x = x_z[:, :, :utils.X_LEN]
            z = x_z[:, :, utils.X_LEN:]
            predicted_x = predicted_x_z[:, :, :utils.X_LEN]
            predicted_z = predicted_x_z[:, :, utils.X_LEN:]

            def velocity(v):
                return (v[:, 1:] - v[:, :-1]) * fps
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

            if rolling_loss is None:
                rolling_loss = loss.item()
            else:
                rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01

            # logging
            if epoch % 10 == 0:
                sys.stdout.write('\rIter: %7i Loss: %5.3f' % (epoch, rolling_loss))

            if epoch % 1000 == 0:
                scheduler.step()

        sys.stdout.write('\n')
        return stepper
