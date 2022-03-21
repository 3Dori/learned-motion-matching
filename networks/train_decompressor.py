import sys

from common.visualization import generate_animation
from networks.learned_motion_AE import Compressor, Decompressor
from networks.utils import extract_locomotion_from_y_feature_vector, randomly_pick_from_dataset
from common.quaternion import from_xy, fk_vel
import common.locomotion_utils as utils

import numpy as np
import torch


class DecompressorTrainer(object):
    def __init__(self, compressor_hidden_size=512, decompressor_hidden_size=512):
        self.compressor_hidden_size = compressor_hidden_size
        self.decompressor_hidden_size = decompressor_hidden_size

    @staticmethod
    def prepare_batches(dataset, sequences, window, batch_size):
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

        for i, action in randomly_pick_from_dataset(dataset, sequences, batch_size):
            frame = np.random.randint(0, action['n_frames'] - 2)
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
        assert i == batch_size
        yield (buffer_x, buffer_y, buffer_q,
               buffer_y_gnd_pos, buffer_y_gnd_txy, buffer_y_gnd_vel, buffer_y_gnd_ang, buffer_y_gnd_rvel,
               buffer_y_gnd_rang,
               buffer_q_gnd_pos, buffer_q_gnd_vel, buffer_q_gnd_ang, buffer_q_gnd_xfm)

    def train(
            self, dataset, batch_size=32, window=2, epochs=10000, lr=0.001,
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
            w_vreg=0.01,
            seed=0
    ):
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

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        rolling_loss = None
        parents = dataset.skeleton().parents()
        fps = dataset.fps()

        sequences = []
        for subject in dataset.subjects():
            for action in dataset[subject].keys():
                sequences.append((subject, action))
        sequences = np.array(sequences)
        batches = self.prepare_batches(dataset, sequences, window, batch_size)

        np.random.seed(seed)
        torch.manual_seed(seed)

        Y_mean = torch.tensor(dataset.Y_mean, dtype=torch.float32).to(device)
        Y_decompressor_scale = torch.tensor(dataset.Y_decompressor_scale, dtype=torch.float32).to(device)

        for epoch in range(epochs):
            optimizer.zero_grad()

            (x, y, q,
             Ygnd_pos, Ygnd_txy, Ygnd_vel, Ygnd_ang, Ygnd_rvel, Ygnd_rang,
             Qgnd_pos, Qgnd_vel, Qgnd_ang, Qgnd_xfm) = next(batches)
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)
            q = torch.tensor(q).to(device)
            Ygnd_pos = torch.tensor(Ygnd_pos).to(device)
            Ygnd_txy = torch.tensor(Ygnd_txy).to(device)
            Ygnd_vel = torch.tensor(Ygnd_vel).to(device)
            Ygnd_ang = torch.tensor(Ygnd_ang).to(device)
            Ygnd_rvel = torch.tensor(Ygnd_rvel).to(device)
            Ygnd_rang = torch.tensor(Ygnd_rang).to(device)
            Qgnd_pos = torch.tensor(Qgnd_pos).to(device)
            Qgnd_vel = torch.tensor(Qgnd_vel).to(device)
            Qgnd_ang = torch.tensor(Qgnd_ang).to(device)
            Qgnd_xfm = torch.tensor(Qgnd_xfm).to(device)

            # encode
            z = compressor((torch.cat([y, q], dim=-1)))    # y and q have been normalized
            # decode
            y_out = decompressor((torch.cat([x, z], dim=-1))) * Y_decompressor_scale + Y_mean

            Ypos, Ytxy, Yvel, Yang, Yrvel, Yrang = extract_locomotion_from_y_feature_vector(y_out, batch_size, window)
            Ypos = torch.cat([Ygnd_pos[:, :, 0:1], Ypos], dim=2)
            Ytxy = torch.cat([Ygnd_txy[:, :, 0:1], Ytxy], dim=2)
            Yvel = torch.cat([Ygnd_vel[:, :, 0:1], Yvel], dim=2)
            Yang = torch.cat([Ygnd_ang[:, :, 0:1], Yang], dim=2)

            # forward kinematics
            Yxfm = from_xy(Ytxy)
            Qxfm, Qpos, Qvel, Qang = fk_vel(Yxfm, Ypos, Yvel, Yang, parents)

            # compute deltas
            Ygnd_dpos = (Ygnd_pos[:, 1:] - Ygnd_pos[:, :-1]) * fps
            Ygnd_drot = (Ygnd_txy[:, 1:] - Ygnd_txy[:, :-1]) * fps
            Qgnd_dpos = (Qgnd_pos[:, 1:] - Qgnd_pos[:, :-1]) * fps
            Qgnd_drot = (Qgnd_xfm[:, 1:] - Qgnd_xfm[:, :-1]) * fps

            Ydpos = (Ypos[:, 1:] - Ypos[:, :-1]) * fps
            Ydrot = (Ytxy[:, 1:] - Ytxy[:, :-1]) * fps
            Qdpos = (Qpos[:, 1:] - Qpos[:, :-1]) * fps
            Qdrot = (Qxfm[:, 1:] - Qxfm[:, :-1]) * fps

            dz = (z[:, 1:] - z[:, :-1]) * fps

            # compute losses
            loss_loc_pos = torch.mean(torch.abs(Ygnd_pos - Ypos))
            loss_loc_txy = torch.mean(torch.abs(Ygnd_txy - Ytxy))
            loss_loc_vel = torch.mean(torch.abs(Ygnd_vel - Yvel))
            loss_loc_ang = torch.mean(torch.abs(Ygnd_ang - Yang))
            loss_loc_rvel = torch.mean(torch.abs(Ygnd_rvel - Yrvel))
            loss_loc_rang = torch.mean(torch.abs(Ygnd_rang - Yrang))

            loss_chr_pos = torch.mean(torch.abs(Qgnd_pos - Qpos))
            loss_chr_xfm = torch.mean(torch.abs(Qgnd_xfm - Qxfm))
            loss_chr_vel = torch.mean(torch.abs(Qgnd_vel - Qvel))
            loss_chr_ang = torch.mean(torch.abs(Qgnd_ang - Qang))

            loss_lvel_pos = torch.mean(torch.abs(Ygnd_dpos - Ydpos))
            loss_lvel_rot = torch.mean(torch.abs(Ygnd_drot - Ydrot))
            loss_cvel_pos = torch.mean(torch.abs(Qgnd_dpos - Qdpos))
            loss_cvel_rot = torch.mean(torch.abs(Qgnd_drot - Qdrot))

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

            # logging
            if rolling_loss is None:
                rolling_loss = loss.item()
            else:
                rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01

            if epoch % 10 == 0:
                sys.stdout.write('\rIter: %7i Loss: %5.3f' % (epoch, rolling_loss))

            if epoch % 1000 == 0:
                generate_animation(dataset, dataset['S1']['jog_1_d0'], compressor, decompressor, device)

            if epoch % 1000 == 0:
                scheduler.step()
