from networks.base_trainer import BaseTrainer
from networks.learned_motion_AE import Compressor, Decompressor
from networks.utils import extract_locomotion_from_y_feature_vector, all_sequences_of_dataset
from common.quaternion import from_xy, fk_vel
import common.locomotion_utils as utils

import numpy as np
import torch

import sys


class DecompressorTrainer(BaseTrainer):
    def __init__(self, compressor_hidden_size=512, decompressor_hidden_size=512):
        super().__init__()
        self.compressor_hidden_size = compressor_hidden_size
        self.decompressor_hidden_size = decompressor_hidden_size

    @staticmethod
    def prepare_batches(dataset, sequences, batch_size, window):
        buffer_x = np.zeros((batch_size, window, utils.X_LEN), dtype=np.float32)
        buffer_y = np.zeros((batch_size, window, utils.Y_LEN), dtype=np.float32)
        buffer_q = np.zeros((batch_size, window, utils.Q_LEN), dtype=np.float32)
        # buffer for root (ground) locomotion
        buffer_y_gnd_pos = np.zeros((batch_size, window, utils.N_BONES, 3), dtype=np.float32)
        buffer_y_gnd_txy = np.zeros((batch_size, window, utils.N_BONES, 3, 2), dtype=np.float32)
        buffer_y_gnd_vel = np.zeros((batch_size, window, utils.N_BONES, 3), dtype=np.float32)
        buffer_y_gnd_ang = np.zeros((batch_size, window, utils.N_BONES, 3), dtype=np.float32)
        buffer_y_gnd_rvel = np.zeros((batch_size, window, 3), dtype=np.float32)
        buffer_y_gnd_rang = np.zeros((batch_size, window, 3), dtype=np.float32)

        buffer_q_gnd_pos = np.zeros((batch_size, window, utils.N_BONES, 3), dtype=np.float32)
        buffer_q_gnd_vel = np.zeros((batch_size, window, utils.N_BONES, 3), dtype=np.float32)
        buffer_q_gnd_ang = np.zeros((batch_size, window, utils.N_BONES, 3), dtype=np.float32)
        buffer_q_gnd_xfm = np.zeros((batch_size, window, utils.N_BONES, 3, 3), dtype=np.float32)

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
                buffer_x[i, 0:window] = action['input_feature'][frame:frame+window]
                buffer_y[i, 0:window] = action['Y_feature'][frame:frame+window]
                buffer_q[i, 0:window] = action['Q_feature'][frame:frame+window]

                buffer_y_gnd_pos[i, 0:window] = action['positions_local'][frame:frame+window]
                buffer_y_gnd_txy[i, 0:window] = action['rotations_txy'][frame:frame+window]
                buffer_y_gnd_vel[i, 0:window] = action['velocities_local'][frame:frame+window]
                buffer_y_gnd_ang[i, 0:window] = action['angular_velocity'][frame:frame+window]
                buffer_y_gnd_rvel[i, 0:window] = action['Yrvel'][frame:frame+window]
                buffer_y_gnd_rang[i, 0:window] = action['Yrang'][frame:frame+window]

                buffer_q_gnd_pos[i, 0:window] = action['Qpos'][frame:frame+window]
                buffer_q_gnd_vel[i, 0:window] = action['Qvel'][frame:frame+window]
                buffer_q_gnd_ang[i, 0:window] = action['Qang'][frame:frame+window]
                buffer_q_gnd_xfm[i, 0:window] = action['Qxfm'][frame:frame+window]
            yield (buffer_x, buffer_y, buffer_q,
                   buffer_y_gnd_pos, buffer_y_gnd_txy, buffer_y_gnd_vel, buffer_y_gnd_ang, buffer_y_gnd_rvel,
                   buffer_y_gnd_rang,
                   buffer_q_gnd_pos, buffer_q_gnd_vel, buffer_q_gnd_ang, buffer_q_gnd_xfm)

    @staticmethod
    def get_compressor_mean_and_scale(dataset, device):
        y_mean = torch.as_tensor(dataset.Y_mean, dtype=torch.float32, device=device)
        q_mean = torch.as_tensor(dataset.Q_mean, dtype=torch.float32, device=device)
        compressor_mean = torch.cat([y_mean, q_mean], dim=-1)

        y_compressor_scale = torch.as_tensor(dataset.Y_compressor_scale, dtype=torch.float32, device=device)
        q_compressor_scale = torch.as_tensor(dataset.Q_compressor_scale, dtype=torch.float32, device=device)
        compressor_scale = torch.cat([y_compressor_scale, q_compressor_scale], dim=-1)

        return compressor_mean, compressor_scale

    def train(
            self, dataset, batch_size=32, window=2, epochs=100000, lr=0.001, seed=0,
            w_loc_pos=75.0,
            w_loc_txy=10.0,
            w_loc_vel=10.0,
            w_loc_ang=1.25,
            w_loc_rvel=2.0,
            w_loc_rang=2.0,
            w_chr_pos=15.0,
            w_chr_xfm=5.0,
            w_chr_vel=2.0,
            w_chr_ang=0.75,
            w_lvel_pos=10.0,
            w_lvel_rot=1.75,
            w_cvel_pos=2.0,
            w_cvel_rot=0.75,
            w_sreg=0.1,
            w_lreg=0.1,
            w_vreg=0.01
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)

        compressor = Compressor(input_size=utils.Y_LEN + utils.Q_LEN,
                                output_size=utils.Z_LEN,
                                hidden_size=self.compressor_hidden_size)
        decompressor = Decompressor(input_size=utils.X_LEN + utils.Z_LEN,
                                    output_size=utils.Y_LEN,
                                    hidden_size=self.decompressor_hidden_size)
        optimizer = torch.optim.AdamW(
            list(compressor.parameters()) +
            list(decompressor.parameters()),
            lr=lr,
            amsgrad=True,
            weight_decay=0.001
        )

        device = dataset.device()
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        parents = dataset.skeleton().parents()
        fps = dataset.fps()

        sequences = all_sequences_of_dataset(dataset)
        batches = self.prepare_batches(dataset, sequences, batch_size, window)

        Y_decompressor_scale = torch.as_tensor(dataset.Y_decompressor_scale, dtype=torch.float32, device=device)
        compressor_mean, compressor_scale = self.get_compressor_mean_and_scale(dataset, device)

        rolling_loss = None
        for epoch in range(epochs):
            optimizer.zero_grad()

            (x, y, q,
             y_gnd_pos, y_gnd_txy, y_gnd_vel, y_gnd_ang, y_gnd_rvel, y_gnd_rang,
             q_gnd_pos, q_gnd_vel, q_gnd_ang, q_gnd_xfm) = next(batches)
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)
            q = torch.tensor(q).to(device)
            y_gnd_pos = torch.tensor(y_gnd_pos).to(device)
            y_gnd_txy = torch.tensor(y_gnd_txy).to(device)
            y_gnd_vel = torch.tensor(y_gnd_vel).to(device)
            y_gnd_ang = torch.tensor(y_gnd_ang).to(device)
            y_gnd_rvel = torch.tensor(y_gnd_rvel).to(device)
            y_gnd_rang = torch.tensor(y_gnd_rang).to(device)
            q_gnd_pos = torch.tensor(q_gnd_pos).to(device)
            q_gnd_vel = torch.tensor(q_gnd_vel).to(device)
            q_gnd_ang = torch.tensor(q_gnd_ang).to(device)
            q_gnd_xfm = torch.tensor(q_gnd_xfm).to(device)

            # encode
            z = compressor((torch.cat([y, q], dim=-1) - compressor_mean) / compressor_scale)
            # decode
            y_out = decompressor((torch.cat([x, z], dim=-1))) * Y_decompressor_scale + Y_mean

            y_pos, y_txy, y_vel, y_ang, y_rvel, y_rang = extract_locomotion_from_y_feature_vector(y_out, batch_size, window)
            y_pos = torch.cat([y_gnd_pos[:, :, 0:1], y_pos], dim=2)
            y_txy = torch.cat([y_gnd_txy[:, :, 0:1], y_txy], dim=2)
            y_vel = torch.cat([y_gnd_vel[:, :, 0:1], y_vel], dim=2)
            y_ang = torch.cat([y_gnd_ang[:, :, 0:1], y_ang], dim=2)

            # forward kinematics
            y_xfm = from_xy(y_txy)
            q_xfm, q_pos, q_vel, q_ang = fk_vel(y_xfm, y_pos, y_vel, y_ang, parents)

            # compute deltas
            y_gnd_dpos = (y_gnd_pos[:, 1:] - y_gnd_pos[:, :-1]) * fps
            y_gnd_drot = (y_gnd_txy[:, 1:] - y_gnd_txy[:, :-1]) * fps
            q_gnd_dpos = (q_gnd_pos[:, 1:] - q_gnd_pos[:, :-1]) * fps
            q_gnd_drot = (q_gnd_xfm[:, 1:] - q_gnd_xfm[:, :-1]) * fps

            y_dpos = (y_pos[:, 1:] - y_pos[:, :-1]) * fps
            y_drot = (y_txy[:, 1:] - y_txy[:, :-1]) * fps
            q_dpos = (q_pos[:, 1:] - q_pos[:, :-1]) * fps
            q_drot = (q_xfm[:, 1:] - q_xfm[:, :-1]) * fps

            dz = (z[:, 1:] - z[:, :-1]) * fps

            # compute losses
            loss_loc_pos = torch.mean(torch.abs(y_gnd_pos - y_pos))
            loss_loc_txy = torch.mean(torch.abs(y_gnd_txy - y_txy))
            loss_loc_vel = torch.mean(torch.abs(y_gnd_vel - y_vel))
            loss_loc_ang = torch.mean(torch.abs(y_gnd_ang - y_ang))
            loss_loc_rvel = torch.mean(torch.abs(y_gnd_rvel - y_rvel))
            loss_loc_rang = torch.mean(torch.abs(y_gnd_rang - y_rang))

            loss_chr_pos = torch.mean(torch.abs(q_gnd_pos - q_pos))
            loss_chr_xfm = torch.mean(torch.abs(q_gnd_xfm - q_xfm))
            loss_chr_vel = torch.mean(torch.abs(q_gnd_vel - q_vel))
            loss_chr_ang = torch.mean(torch.abs(q_gnd_ang - q_ang))

            loss_lvel_pos = torch.mean(torch.abs(y_gnd_dpos - y_dpos))
            loss_lvel_rot = torch.mean(torch.abs(y_gnd_drot - y_drot))
            loss_cvel_pos = torch.mean(torch.abs(q_gnd_dpos - q_dpos))
            loss_cvel_rot = torch.mean(torch.abs(q_gnd_drot - q_drot))

            loss_sreg = torch.mean(torch.abs(z))
            loss_lreg = torch.mean(torch.square(z))
            loss_vreg = torch.mean(torch.abs(dz))

            loss = (
                loss_loc_pos * w_loc_pos +
                loss_loc_txy * w_loc_txy +
                loss_loc_vel * w_loc_vel +
                loss_loc_ang * w_loc_ang +
                loss_loc_rvel * w_loc_rvel +
                loss_loc_rang * w_loc_rang +

                loss_chr_pos * w_chr_pos +
                loss_chr_xfm * w_chr_xfm +
                loss_chr_vel * w_chr_vel +
                loss_chr_ang * w_chr_ang +

                loss_lvel_pos * w_lvel_pos +
                loss_lvel_rot * w_lvel_rot +
                loss_cvel_pos * w_cvel_pos +
                loss_cvel_rot * w_cvel_rot +

                loss_sreg * w_sreg +
                loss_lreg * w_lreg +
                loss_vreg * w_vreg
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
        return compressor, decompressor
