from networks.learned_motion_AE import Stepper
from networks.utils import randomly_pick_from_dataset, all_sequences_of_dataset
import common.locomotion_utils as utils

import numpy as np
import torch


class StepperTrainer(object):
    def __init__(self, compressor, dataset, hidden_size=512):
        self.compressor = compressor
        self.hidden_size = 512

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
        buffer_xs = [np.zeros((batch_size, utils.X_LEN), dtype=np.float32) for _ in range(window)]
        buffer_zs = [np.zeros((batch_size, utils.Z_LEN), dtype=np.float32) for _ in range(window)]

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
                for w in range(window):
                    buffer_xs[w][i] = action['input_feature'][frame+w]
                    buffer_zs[w][i] = action['Z_code'][frame+w]
            yield buffer_xs, buffer_zs

    def train(self, dataset, batch_size=32, window=2, epochs=10000, lr=0.001):
        stepper = Stepper(param_size=utils.X_LEN + utils.Z_LEN, hidden_size=self.hidden_size)
        optimizer = torch.optim.AdamW(
            stepper.parameters(),
            lr=lr,
            amsgrad=True,
            weight_decay=0.001
        )

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        sequences = all_sequences_of_dataset(dataset)
        batches = self.prepare_batches(dataset, sequences, window, batch_size)
        for epoch in range(epochs):
            optimizer.zero_grad()

            xs, zs = next(batches)
            x = torch.tensor(x).to(device)
            z = torch.tensor(z).to(device)
            x_z_i = torch.cat([x[:, 0], z[:, 0]], dim=-1)
            for i in range(window):
                delta_x_z = stepper(x_z_i)
                x_z_i += delta_x_z

            loss_xval = torch.mean(torch.abs())
            loss_zval = 0
            loss_xvel = 0
            loss_zvel = 0

            loss = (
                loss_xval * w_xval +
                loss_zval * w_zval +
                loss_xvel * w_xvel +
                loss_zvel * w_zvel
            )
