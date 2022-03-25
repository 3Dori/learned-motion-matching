# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle

import scipy.ndimage.filters
import scipy.spatial
import numpy as np
import torch

from common.spline import Spline
from common.quaternion import angular_velocity_np, to_scaled_angle_axis_np, to_xform_xy_np, qrot_inv_np, fk_vel_np, \
    to_xform_np

_use_gpu = torch.cuda.is_available()

lf = 4    # Left foot index
rf = 9    # Right foot index
hip = 0   # Hip index

N_BONES = 26

X_LEN = (2 * 3 +    # future trajectory at ground, at 20, 40, 60 frames ahead
         2 * 3 +    # future facing direction, at 20, 40, 60 frames ahead
         3 * 2 +    # left foot local position
         3 * 2 +    # right foot local position
         1)         # hip y-velocity

X_OFFSETS = [0, 6, 12, 18, 24, 25]
X_WEIGHTS = [1, 1, 1, 1, 1]

Y_LEN = ((N_BONES-1) * 3 +
         (N_BONES-1) * 6 +
         (N_BONES-1) * 3 +
         (N_BONES-1) * 3 +
         3 +
         3)

Z_LEN = 512

Y_OFFSETS = [0, 75, 225, 300, 375, 378, 381]

Q_LEN = ((N_BONES-1) * 3 +
         (N_BONES-1) * 6 +
         (N_BONES-1) * 3 +
         (N_BONES-1) * 3)

Q_OFFSETS = [0, 75, 225, 300, 375]


def build_phase_track(positions_world):
    """
    Detect foot steps and extract a control signal that describes the current state of the walking cycle.
    This is based on the assumption that the speed of a foot is almost zero during a contact.
    """
    l_speed = np.linalg.norm(np.diff(positions_world[:, lf], axis=0), axis=1)
    r_speed = np.linalg.norm(np.diff(positions_world[:, rf], axis=0), axis=1)
    displacements = np.cumsum(np.linalg.norm(np.diff(positions_world[:, 0], axis=0), axis=1))
    left_contact = l_speed[0] < r_speed[0]
    epsilon = 0.1 # Hysteresis (i.e. minimum height difference before a foot switch is triggered)
    cooldown = 3 # Minimum # of frames between steps
    accumulator = np.pi if left_contact else 0
    phase_points = [ (0, accumulator) ]
    disp_points = [ (0, displacements[0]) ]
    i = cooldown
    while i < len(l_speed):
        if left_contact and l_speed[i] > r_speed[i] + epsilon:
            left_contact = False
            accumulator += np.pi
            phase_points.append((i, accumulator))
            disp_points.append((i, displacements[i] - displacements[disp_points[-1][0]]))
            i += cooldown
        elif not left_contact and r_speed[i] > l_speed[i] + epsilon:
            left_contact = True
            accumulator += np.pi
            phase_points.append((i, accumulator))
            disp_points.append((i, displacements[i] - displacements[disp_points[-1][0]]))
            i += cooldown
        else:
            i += 1
            
    phase = np.zeros(l_speed.shape[0])
    end_idx = 0
    for i in range(len(phase_points) - 1):
        start_idx = phase_points[i][0]
        end_idx = phase_points[i+1][0]
        phase[start_idx:end_idx] = np.linspace(phase_points[i][1], phase_points[i+1][1], end_idx-start_idx, endpoint=False)
    phase[end_idx:] = phase_points[-1][1]
    last_point = (phase[-1] - phase[-2]) + phase[-1]
    phase = np.concatenate((phase, [last_point]))
    return phase


def extract_translations_controls(positions_world, xy_orientation):
    """
    Extract extra features for the long-term network:
    - Translations: longitudinal speed along the spline; height of the root joint.
    - Controls: walking phase as a [cos(theta), sin(theta)] signal; same for the facing direction and movement direction.
    """
    phase = build_phase_track(positions_world)
    xy = np.diff(positions_world[:, 0, [0, 2]], axis=0)    # root velocity along the xy axis
    xy = np.concatenate((xy, xy[-1:]), axis=0)
    z = positions_world[:, 0, 1] # We use a xzy coordinate system
    speeds_abs = np.linalg.norm(xy, axis=1) # Instantaneous speed along the trajectory
    amplitude = scipy.ndimage.filters.gaussian_filter1d(speeds_abs, 5) # Low-pass filter
    speeds_abs -= amplitude # Extract high-frequency details
    # Integrate the high-frequency speed component to recover an offset w.r.t. the trajectory
    speeds_abs = np.cumsum(speeds_abs)
    
    xy /= np.linalg.norm(xy, axis=1).reshape(-1, 1) + 1e-9 # Epsilon to avoid division by zero

    return np.stack((speeds_abs, z, # Translations
                     np.cos(phase)*amplitude, np.sin(phase)*amplitude, # Controls
                     xy[:, 0]*amplitude, xy[:, 1]*amplitude, # Controls
                     np.sin(xy_orientation), np.cos(xy_orientation)), # Controls
                    axis=1)


def build_extra_features(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            xy_orientation = dataset[subject][action]['rotations_euler'][:, 0, 1]
            dataset[subject][action]['extra_features'] = extract_translations_controls(
                                                                dataset[subject][action]['positions_world'],
                                                                xy_orientation)


def compute_bone_positions(action, skeleton):
    rotations = torch.from_numpy(action['rotations'].astype('float32')).unsqueeze(0)
    trajectory = torch.from_numpy(action['trajectory'].astype('float32')).unsqueeze(0)
    if _use_gpu:
        rotations = rotations.cuda()
        trajectory = trajectory.cuda()
    action['positions_world'] = skeleton.forward_kinematics(rotations, trajectory).squeeze(0).cpu().numpy()

    # Absolute translations across the XY plane are removed here
    trajectory[:, :, [0, 2]] = 0
    action['positions_local'] = skeleton.forward_kinematics(rotations, trajectory).squeeze(0).cpu().numpy()


def compute_bone_velocities(action, fps):
    action['velocities_local'] = np.diff(action['positions_local'], axis=0) * fps
    action['velocities_world'] = np.diff(action['positions_world'], axis=0) * fps
    action['angular_velocity'] = to_scaled_angle_axis_np(angular_velocity_np(action['rotations'], fps))
    action['rotations_txy'] = to_xform_xy_np(action['rotations'])

    for feature in ('velocities_local', 'velocities_world', 'angular_velocity'):
        action[feature] = np.concatenate((action[feature], action[feature][-1:]), axis=0)

    assert(len(action['rotations']) == len(action['trajectory']) == len(action['angular_velocity']) ==
           len(action['positions_world']) == len(action['positions_local']) ==
           len(action['velocities_world']) == len(action['velocities_local']))


def _future_trajectory_at_frame(positions_world, frame):
    frame = min(positions_world.shape[0], frame)
    return np.vstack((positions_world[frame:, 0, [0, 2]],  # only x and z values
                      np.repeat(positions_world[-1:, 0, [0, 2]], frame, axis=0)))


def _future_facing_direction_at_frame(facing_direction, frame):
    frame = min(facing_direction.shape[0], frame)
    return np.vstack((facing_direction[frame:],
                      np.repeat(facing_direction[-1:], frame, axis=0)))


def _build_input_feature_vector_for_action(action):
    input_feature = np.zeros((action['n_frames'], X_LEN), dtype=np.float32)

    # future trajectory
    positions_world = action['positions_world']

    input_feature[:, 0:2] = _future_trajectory_at_frame(positions_world, 20)
    input_feature[:, 2:4] = _future_trajectory_at_frame(positions_world, 40)
    input_feature[:, 4:6] = _future_trajectory_at_frame(positions_world, 60)

    # future facing direction
    xy_orientation = action['rotations_euler'][:, 0, 1]
    facing_direction = np.stack((np.sin(xy_orientation), np.cos(xy_orientation)), axis=1)  # to be normalized?
    input_feature[:, 6:8] = _future_facing_direction_at_frame(facing_direction, 20)
    input_feature[:, 8:10] = _future_facing_direction_at_frame(facing_direction, 40)
    input_feature[:, 10:12] = _future_facing_direction_at_frame(facing_direction, 60)

    # foot position
    input_feature[:, 12:15] = action['positions_local'][:, lf]
    input_feature[:, 15:18] = action['positions_local'][:, rf]
    # foot velocity
    input_feature[:, 18:21] = action['velocities_local'][:, lf]
    input_feature[:, 21:24] = action['velocities_local'][:, rf]
    # hip y-velocity
    input_feature[:, 24] = action['velocities_local'][:, hip, 1]

    return input_feature


def _std(mean, sum2, n):
    std = np.sqrt(sum2 / n - mean ** 2)
    return std


def _std_with_range(sum_, sum2, n, start, end):
    sum_ = sum_[start:end].sum()
    sum2 = sum2[start:end].sum()
    mean = sum_ / (n * (end - start))
    std = np.sqrt(sum2 / (n * (end - start)) - mean ** 2)
    return std


def compute_input_features(dataset):
    # TODO: use mean and scale for each action
    dataset.decompressor_mean = np.zeros(X_LEN)
    dataset.decompressor_scale = np.zeros(X_LEN)
    for subject in dataset.subjects():
        for action in dataset[subject].values():
            # Compute positions and velocities
            if 'positions_world' not in action or 'positions_local' not in action:
                compute_bone_positions(action, dataset.skeleton())

            if 'velocities_world' not in action or 'velocities_local' not in action or 'angular_velocity' not in action:
                compute_bone_velocities(action, dataset.fps())

            action['input_feature'] = _build_input_feature_vector_for_action(action)

    # normalize input features
    input_sum = np.zeros(X_LEN)
    input_sum2 = np.zeros(X_LEN)
    for subject in dataset.subjects():
        for action in dataset[subject].values():
            input_sum += action['input_feature'].sum(axis=0)
            input_sum2 += (action['input_feature'] ** 2).sum(axis=0)
    dataset.decompressor_mean = input_sum / dataset.n_total_frames
    input_std = _std(mean=dataset.decompressor_mean, sum2=input_sum2, n=dataset.n_total_frames)

    for start, end, weight in zip(X_OFFSETS, X_OFFSETS[1:], X_WEIGHTS):
        std = input_std[start:end].mean()
        dataset.decompressor_scale[start:end] = std / weight


def compute_z_vector(dataset, compressor):
    # generate z features
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for subject in dataset.subjects():
            for action in dataset[subject].values():
                y = torch.tensor(action['Y_feature'][np.newaxis]).to(device)
                q = torch.tensor(action['Q_feature'][np.newaxis]).to(device)

                z = compressor(torch.cat([y, q], dim=-1))
                action['Z_code'] = z.reshape((-1, Z_LEN)).cpu().numpy()
    # normalize
    dataset.stepper_mean_in = np.zeros(X_LEN + Z_LEN, dtype=np.float32)
    dataset.stepper_mean_out = np.zeros(X_LEN + Z_LEN, dtype=np.float32)
    dataset.stepper_std_in = np.zeros(X_LEN + Z_LEN, dtype=np.float32)
    dataset.stepper_std_out = np.zeros(X_LEN + Z_LEN, dtype=np.float32)

    fps = dataset.fps()
    n_actions = len(dataset.all_actions())
    x_sum = np.zeros(X_LEN)
    z_sum = np.zeros(Z_LEN)
    x_sum2 = np.zeros(X_LEN)
    z_sum2 = np.zeros(Z_LEN)
    dx_sum = np.zeros(X_LEN)
    dz_sum = np.zeros(Z_LEN)
    dx_sum2 = np.zeros(X_LEN)
    dz_sum2 = np.zeros(Z_LEN)
    for subject in dataset.subjects():
        for action in dataset[subject].values():
            x_sum += action['input_feature'].sum(axis=0)
            x_sum2 += (action['input_feature'] ** 2).sum(axis=0)
            z_sum += action['Z_code'].sum(axis=0)
            z_sum2 += (action['Z_code'] ** 2).sum(axis=0)

            dx = (action['input_feature'][1:] - action['input_feature'][:-1]) * fps
            dx_sum += dx.sum(axis=0)
            dx_sum2 += (dx ** 2).sum(axis=0)
            dz = (action['Z_code'][1:] - action['Z_code'][:-1]) * fps
            dz_sum += dz.sum(axis=0)
            dz_sum2 += (dz ** 2).sum(axis=0)
    dataset.stepper_mean_in[:X_LEN] = x_sum / dataset.n_total_frames
    dataset.stepper_std_in[:X_LEN] = _std_with_range(x_sum, x_sum2, dataset.n_total_frames, 0, X_LEN)
    dataset.stepper_mean_in[X_LEN:] = z_sum / dataset.n_total_frames
    dataset.stepper_std_in[X_LEN:] = _std_with_range(z_sum, z_sum2, dataset.n_total_frames, X_LEN, X_LEN + Z_LEN)

    dataset.stepper_mean_out[:X_LEN] = dx_sum / (dataset.n_total_frames - n_actions)
    dataset.stepper_std_out[:X_LEN] = _std(dataset.stepper_mean_out[:X_LEN], dx_sum2, (dataset.n_total_frames - n_actions))
    dataset.stepper_mean_out[X_LEN:] = dz_sum / (dataset.n_total_frames - n_actions)
    dataset.stepper_std_out[X_LEN:] = _std(dataset.stepper_mean_out[X_LEN:], dz_sum2, (dataset.n_total_frames - n_actions))

    dataset.projector_mean_in = dataset.stepper_mean_in[:X_LEN]
    dataset.projector_std_in = dataset.stepper_std_in[:X_LEN]
    dataset.projector_mean_out = dataset.stepper_mean_in.copy()
    dataset.projector_std_out = dataset.stepper_std_in.copy()

    with open('datasets/dataset_learned_motion.pkl', 'wb') as f:
        pickle.dump(dataset, f)


def nearest_frame(dataset, x):
    # return z and x vectors of that frame, instead of the frame index
    # return shape (batch_size, X_LEN), (batch_size, Z_LEN)
    batch_size = x.shape[0]
    min_dist = np.ones(batch_size, dtype=np.float32) * np.inf
    nearest_x = np.zeros((batch_size, X_LEN), dtype=np.float32)
    nearest_z = np.zeros((batch_size, Z_LEN), dtype=np.float32)
    for subject in dataset.subjects():
        for action in dataset[subject].values():
            dist = scipy.spatial.distance_matrix(x, action['input_feature'])    # batch_size * n_frames
            local_min_idx = np.argmin(dist, axis=1)
            local_min_dist = np.min(dist, axis=1)
            nearest_x[local_min_dist < min_dist] = action['input_feature'][local_min_idx]
            nearest_z[local_min_dist < min_dist] = action['Z_code'][local_min_idx]
    return nearest_x, nearest_z


def _build_output_vector_for_action(action, parents):
    n_frames = action['n_frames']
    y_feature = np.zeros((n_frames, Y_LEN), dtype=np.float32)
    q_feature = np.zeros((n_frames, Q_LEN), dtype=np.float32)

    y_pos = action['positions_local']
    y_rot = action['rotations']
    y_txy = action['rotations_txy']
    y_vel = action['velocities_local']
    y_ang = action['angular_velocity']
    y_rvel = qrot_inv_np(y_rot[:, 0], y_vel[:, 0])
    y_rang = qrot_inv_np(y_rot[:, 0], y_ang[:, 0])
    action['Yrvel'] = y_rvel
    action['Yrang'] = y_rang

    y_feature[:, 0:75] = y_pos[:, 1:].reshape((n_frames, -1))
    y_feature[:, 75:225] = y_txy[:, 1:].reshape((n_frames, -1))
    y_feature[:, 225:300] = y_vel[:, 1:].reshape((n_frames, -1))
    y_feature[:, 300:375] = y_ang[:, 1:].reshape((n_frames, -1))
    y_feature[:, 375:378] = y_rvel.reshape((n_frames, -1))
    y_feature[:, 378:381] = y_rang.reshape((n_frames, -1))

    # Q
    q_rot, q_pos, q_vel, q_ang = fk_vel_np(y_rot, y_pos, y_vel, y_ang, parents)
    q_xfm = to_xform_np(q_rot)
    # Qtxy = to_xform_xy_np(Qrot)
    q_txy = q_xfm[:, :, :, :2]
    action['Qrot'] = q_rot.astype(np.float32)
    action['Qpos'] = q_pos.astype(np.float32)
    action['Qvel'] = q_vel.astype(np.float32)
    action['Qang'] = q_ang.astype(np.float32)
    action['Qtxy'] = q_txy.astype(np.float32)
    action['Qxfm'] = q_xfm.astype(np.float32)

    q_feature[:, 0:75] = q_pos[:, 1:].reshape((n_frames, -1))
    q_feature[:, 75:225] = q_txy[:, 1:].reshape((n_frames, -1))
    q_feature[:, 225:300] = q_vel[:, 1:].reshape((n_frames, -1))
    q_feature[:, 300:375] = q_ang[:, 1:].reshape((n_frames, -1))

    return y_feature, q_feature


def compute_output_features(dataset):
    parents = dataset.skeleton().parents()
    for subject in dataset.subjects():
        for action in dataset[subject].values():
            action['Y_feature'], action['Q_feature'] = _build_output_vector_for_action(action, parents)

    dataset.Y_compressor_scale = np.zeros(Y_LEN)
    dataset.Q_compressor_scale = np.zeros(Q_LEN)
    # normalize
    y_sum = np.zeros(Y_LEN)
    y_sum2 = np.zeros(Y_LEN)
    q_sum = np.zeros(Q_LEN)
    q_sum2 = np.zeros(Q_LEN)
    for subject in dataset.subjects():
        for action in dataset[subject].values():
            y_sum += action['Y_feature'].sum(axis=0)
            y_sum2 += (action['Y_feature'] ** 2).sum(axis=0)
            q_sum += action['Q_feature'].sum(axis=0)
            q_sum2 += (action['Q_feature'] ** 2).sum(axis=0)
    dataset.Y_mean = y_sum / dataset.n_total_frames
    dataset.Q_mean = q_sum / dataset.n_total_frames
    for start, end in zip(Y_OFFSETS, Y_OFFSETS[1:]):
        std = _std_with_range(y_sum, y_sum2, dataset.n_total_frames, start, end)
        dataset.Y_compressor_scale[start:end] = std
    for start, end in zip(Q_OFFSETS, Q_OFFSETS[1:]):
        std = _std_with_range(q_sum, q_sum2, dataset.n_total_frames, start, end)
        dataset.Q_compressor_scale[start:end] = std
    dataset.Y_decompressor_scale = _std(mean=dataset.Y_mean, sum2=y_sum2, n=dataset.n_total_frames).astype(np.float32)
    dataset.Q_decompressor_scale = _std(mean=dataset.Q_mean, sum2=q_sum2, n=dataset.n_total_frames).astype(np.float32)


def angle_difference_batch(y, x):
    """
    Compute the signed angle difference y - x,
    where x and y are given as versors.
    """
    return np.arctan2(y[:, :, 1]*x[:, :, 0] - y[:, :, 0]*x[:, :, 1], y[:, :, 0]*x[:, :, 0] + y[:, :, 1]*x[:, :, 1])


def phase_to_features(phase_signal):
    """
    Given a [A(t)*cos(phase), A(t)*sin(phase)] signal, extract a set of features:
    A(t), absolute phase (not modulo 2*pi), angular velocity.
    This function expects a (batch_size, seq_len, 2) tensor.
    """
    assert len(phase_signal.shape) == 3
    assert phase_signal.shape[-1] == 2
    amplitudes = np.linalg.norm(phase_signal, axis=2).reshape(phase_signal.shape[0], -1, 1)
    phase_signal = phase_signal/(amplitudes + 1e-9)
    phase_signal_diff = angle_difference_batch(phase_signal[:, 1:], phase_signal[:, :-1])
    frequencies = np.pad(phase_signal_diff, ((0, 0), (0, 1)), 'edge').reshape(phase_signal_diff.shape[0], -1, 1)
    return amplitudes, np.cumsum(phase_signal_diff, axis=1), frequencies


def compute_splines(dataset):
    """
    For each animation in the dataset, this method computes its equal-segment-length spline,
    along with its interpolated tracks (e.g. local speed, footstep frequency).
    """
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            xy = dataset[subject][action]['positions_world'][:, 0, [0, 2]]
            spline = Spline(xy, closed=False)

            # Add extra tracks (facing direction, phase/amplitude)
            xy_orientation = dataset[subject][action]['rotations_euler'][:, 0, 1]
            y_rot = np.array([np.sin(xy_orientation), np.cos(xy_orientation)]).T
            spline.add_track('direction', y_rot, interp_mode='circular')
            phase = dataset[subject][action]['extra_features'][:, [2, 3]]
            amplitude, _, frequency = phase_to_features(np.expand_dims(phase, 0))

            spline.add_track('amplitude', amplitude[0], interp_mode='linear')
            spline.add_track('frequency', frequency[0], interp_mode='linear')

            spline = spline.reparameterize(5, smoothing_factor=1)
            avg_speed_track = spline.get_track('amplitude')
            avg_speed_track[:] = np.mean(avg_speed_track)
            spline.add_track('average_speed', avg_speed_track, interp_mode='linear')
            
            dataset[subject][action]['spline'] = spline