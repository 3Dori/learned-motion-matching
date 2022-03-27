# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np
import torch
from pymo.parsers import BVHParser
from pymo.viz_tools import *
from pymo.preprocessing import *
from matplotlib import pyplot as plt

from common.dataset_locomotion import load_dataset
from common.quaternion import qrot_np, qmul_np, from_scaled_angle_axis_np, qeuler_np, from_xform_xy_np, to_euler_np
from networks.decompressor_trainer import DecompressorTrainer
from networks.projector_trainer import ProjectorTrainer
from networks.stepper_trainer import StepperTrainer
from networks.utils import extract_locomotion_from_y_feature_vector, COMPRESSOR_PATH, DECOMPRESSOR_PATH, STEPPER_PATH, \
    PROJECTOR_PATH
from common.locomotion_utils import Y_LEN


def render_Y(y, skeleton, fps, action):
    # Extract required components
    y_pos, y_txy, y_vel, y_ang, y_rvel, y_rang = extract_locomotion_from_y_feature_vector(y, 1)
    n_frames = y_pos.shape[1]

    # Convert to quat and remove batch
    y_rot = from_xform_xy_np(y_txy[0].cpu().numpy())
    y_pos = y_pos[0].cpu().numpy()
    y_rvel = y_rvel[0].cpu().numpy()
    y_rang = y_rang[0].cpu().numpy()

    # Integrate root displacement

    y_root_rot = [action['rotations'][0, 0]]
    y_root_pos = [action['positions_local'][0, 0]]
    for i in range(1, n_frames):
        y_root_pos.append(y_root_pos[-1] + qrot_np(y_root_rot[-1], y_rvel[i - 1]) / fps)
        y_root_rot.append(qmul_np(y_root_rot[-1], from_scaled_angle_axis_np(
            qrot_np(y_root_rot[-1], y_rang[i - 1]) / fps)))

    y_root_pos = np.concatenate([p[np.newaxis] for p in y_root_pos])
    y_pos = np.concatenate([y_root_pos[:, np.newaxis], y_pos], axis=1)

    render_animation(y_pos, skeleton, fps, output='interactive')


def generate_decompressor_animation():
    dataset = load_dataset()
    action = dataset['S1']['jog_1_d0']

    compressor = torch.load(COMPRESSOR_PATH)
    decompressor = torch.load(DECOMPRESSOR_PATH)
    with torch.no_grad():
        device = dataset.device()
        trainer = DecompressorTrainer(dataset)

        y = torch.tensor(action['Y_feature'][np.newaxis]).to(device)
        q = torch.tensor(action['Q_feature'][np.newaxis]).to(device)
        x = torch.tensor(action['input_feature'][np.newaxis]).to(device)

        # Pass through compressor
        z = trainer.compress(compressor, y, q)
        y = trainer.decompress(decompressor, x, z)
        render_Y(y, dataset.skeleton(), dataset.fps(), action)


def generate_stepper_animation(n_frames=360):
    dataset = load_dataset()
    action = dataset['S1']['jog_1_d0']

    stepper = torch.load(STEPPER_PATH)
    decompressor = torch.load(DECOMPRESSOR_PATH)
    with torch.no_grad():
        device = dataset.device()
        trainer = DecompressorTrainer(dataset)

        # compute first
        x_first_frame = torch.as_tensor(action['input_feature'][0:1][np.newaxis], dtype=torch.float32, device=device)
        z_first_frame = torch.as_tensor(action['Z_code'][0:1][np.newaxis], dtype=torch.float32, device=device)
        x_z = torch.cat([x_first_frame, z_first_frame], dim=-1)
        predicted_x_z = StepperTrainer.predict_x_z(stepper, x_z)
        y = trainer.decompress(decompressor, predicted_x_z)

        render_Y(y, dataset.skeleton(), dataset.fps(), action)


def generate_motion_matching_animation(projector_n_frames=10, simulate_n_frames=360):
    dataset = load_dataset()
    action = dataset['S1']['jog_1_d0']
    device = dataset.device()

    decompressor = torch.load(DECOMPRESSOR_PATH)
    stepper = torch.load(STEPPER_PATH)
    projector = torch.load(PROJECTOR_PATH)

    y = torch.zeros((1, simulate_n_frames, Y_LEN), dtype=torch.float32, device=device)

    with torch.no_grad():
        projector_trainer = ProjectorTrainer(dataset)
        stepper_trainer = StepperTrainer(dataset, compressor=None)
        decompressor_trainer = DecompressorTrainer(dataset)

        n_projects = simulate_n_frames // projector_n_frames    # number of projector callings
        x = torch.as_tensor(action['input_feature'][0:1][np.newaxis], dtype=torch.float32, device=device)
        for i in range(n_projects):
            x_z = projector_trainer.project(projector, x)
            predicted_x_z = stepper_trainer.predict_x_z(stepper, x_z, window=projector_n_frames)
            y_out = decompressor_trainer.decompress(decompressor, predicted_x_z)
            y[:, i*projector_n_frames:(i+1)*projector_n_frames] = y_out

        render_Y(y, dataset.skeleton(), dataset.fps(), action)


def render_animation(data, skeleton, fps, output='interactive', bitrate=1000):
    """
    Render or show an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    x = 0
    y = 1
    z = 2
    radius = torch.max(skeleton.offsets()).item() * 5 # Heuristic that works well with many skeletons

    skeleton_parents = skeleton.parents()

    plt.ioff()
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20., azim=30)

    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius/2, radius/2])
    ax.set_aspect('auto')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5

    lines = []
    initialized = False

    trajectory = data[:, 0, [0, 2]]
    avg_segment_length = np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3
    draw_offset = int(25/avg_segment_length)
    spline_line, = ax.plot(*trajectory.T)
    camera_pos = trajectory
    height_offset = np.min(data[:, :, 1]) # Min height
    data = data.copy()
    data[:, :, 1] -= height_offset

    def update(frame):
        nonlocal initialized
        ax.set_xlim3d([-radius/2 + camera_pos[frame, 0], radius/2 + camera_pos[frame, 0]])
        ax.set_ylim3d([-radius/2 + camera_pos[frame, 1], radius/2 + camera_pos[frame, 1]])

        positions_world = data[frame]
        for i in range(positions_world.shape[0]):
            if skeleton_parents[i] == -1:
                continue
            if not initialized:
                col = 'red' if i in skeleton.joints_right() else 'black' # As in audio cables :)
                lines.append(ax.plot([positions_world[i, x], positions_world[skeleton_parents[i], x]],
                        [positions_world[i, y], positions_world[skeleton_parents[i], y]],
                        [positions_world[i, z], positions_world[skeleton_parents[i], z]], zdir='y', c=col))
            else:
                lines[i-1][0].set_xdata(np.array([positions_world[i, x], positions_world[skeleton_parents[i], x]]))
                lines[i-1][0].set_ydata(np.array([positions_world[i, y], positions_world[skeleton_parents[i], y]]))
                lines[i-1][0].set_3d_properties([positions_world[i, z], positions_world[skeleton_parents[i], z]], zdir='y')
        l = max(frame-draw_offset, 0)
        r = min(frame+draw_offset, trajectory.shape[0])
        spline_line.set_xdata(trajectory[l:r, 0])
        spline_line.set_ydata(np.zeros_like(trajectory[l:r, 0]))
        spline_line.set_3d_properties(trajectory[l:r, 1], zdir='y')
        initialized = True
        if output == 'interactive' and frame == data.shape[0] - 1:
            plt.close('all')

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=np.arange(0, data.shape[0]), interval=1000/fps, repeat=False)
    if output == 'interactive':
        plt.show()
        return anim
    elif output == 'html':
        return anim.to_html5_video()
    elif output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only html, .mp4, and .gif are supported)')
    plt.close()


def vis_skeleton(path, out_path_prefix, frames=(0, 10, 20, 30, 40, 50)):
    parser = BVHParser()
    parsed_data = parser.parse(path)
    # parsed_data.skeleton = {k: v for k, v in parsed_data.skeleton.items() if not k.endswith('_Nub')}
    # for skeleton in parsed_data.skeleton.values():
    #     skeleton['children'] = [c for c in skeleton['children'] if not c.endswith('_Nub')]

    mp = MocapParameterizer('position')

    positions = mp.fit_transform([parsed_data])
    # joints_to_visualize = [j for j in parsed_data.skeleton.keys() if not j.startswith("joint_0")]
    joints_to_visualize = parsed_data.skeleton.keys()

    for frame in frames:
        draw_stickfigure(positions[0], frame=frame, joints=joints_to_visualize)
        plt.savefig(out_path_prefix + str(frame))


if __name__ == '__main__':
    # generate_decompressor_animation()
    # generate_stepper_animation()
    generate_motion_matching_animation()
