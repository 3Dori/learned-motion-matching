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

from common.quaternion import qrot_np, qmul_np, from_scaled_angle_axis_np, qeuler_np, from_xform_xy_np, to_euler_np
from networks.utils import extract_locomotion_from_y_feature_vector
import common.locomotion_utils as utils
import common.bvh as bvh


def generate_animation(dataset, action, compressor, decompressor, device):
    with torch.no_grad():
        # Get slice of database for first clip

        # action = dataset['S1']['jog_1_d0']
        fps = dataset.fps()
        parents = dataset.skeleton().parents()
        n_frames = action['n_frames']
        Y_mean = torch.tensor(dataset.Y_mean, dtype=torch.float32).to(device)
        Y_decompressor_scale = torch.tensor(dataset.Y_decompressor_scale, dtype=torch.float32).to(device)

        # Ygnd_rot = torch.tensor(action['rotations'][np.newaxis]).to(device)
        # Ygnd_txy = torch.tensor(action['rotations_txy'][np.newaxis]).to(device)
        # Ygnd_vel = torch.tensor(action['velocities_local'][np.newaxis]).to(device)
        # Ygnd_ang = torch.tensor(action['angular_velocity'][np.newaxis]).to(device)
        #
        # Qgnd_pos = torch.tensor(action['Qpos'][np.newaxis]).to(device)
        # Qgnd_txy = torch.tensor(action['Qtxy'][np.newaxis]).to(device)
        # Qgnd_vel = torch.tensor(action['Qvel'][np.newaxis]).to(device)
        # Qgnd_ang = torch.tensor(action['Qang'][np.newaxis]).to(device)
        #
        # Ygnd_rvel = torch.tensor(action['Yrvel'][np.newaxis]).to(device)
        # Ygnd_rang = torch.tensor(action['Yrang'][np.newaxis]).to(device)

        y = torch.tensor(action['Y_feature'][np.newaxis]).to(device)
        q = torch.tensor(action['Q_feature'][np.newaxis]).to(device)
        x = torch.tensor(action['input_feature'][np.newaxis]).to(device)

        # Pass through compressor
        z = compressor(torch.cat([y, q], dim=-1))

        # Pass through decompressor
        y_out = decompressor((torch.cat([x, z], dim=-1))) * Y_decompressor_scale + Y_mean

        # Extract required components
        Ypos, Ytxy, Yvel, Yang, Yrvel, Yrang = extract_locomotion_from_y_feature_vector(y_out, 1, n_frames)

        # Convert to quat and remove batch
        Yrot = from_xform_xy_np(Ytxy[0].cpu().numpy())
        Ypos = Ypos[0].cpu().numpy()
        Yrvel = Yrvel[0].cpu().numpy()
        Yrang = Yrang[0].cpu().numpy()

        # Integrate root displacement

        Yrootrot = [action['rotations'][0, 0]]
        Yrootpos = [action['positions_local'][0, 0]]
        for i in range(1, n_frames):
            Yrootpos.append(Yrootpos[-1] + qrot_np(Yrootrot[-1], Yrvel[i - 1]) / fps)
            Yrootrot.append(qmul_np(Yrootrot[-1], from_scaled_angle_axis_np(
                qrot_np(Yrootrot[-1], Yrang[i - 1]) / fps)))

        Yrootrot = np.concatenate([r[np.newaxis] for r in Yrootrot])
        Yrootpos = np.concatenate([p[np.newaxis] for p in Yrootpos])

        Yrot = np.concatenate([Yrootrot[:, np.newaxis], Yrot], axis=1)
        Ypos = np.concatenate([Yrootpos[:, np.newaxis], Ypos], axis=1)

        # Write BVH

        try:
            bvh.save('decompressor_Ygnd.bvh', {
                'rotations': np.degrees(to_euler_np(action['rotations'])),
                'positions': 100.0 * action['positions_local'],
                'offsets': 100.0 * action['positions_local'][0],
                'parents': parents,
                'names': ['joint_%i' % i for i in range(utils.N_BONES)],
                'order': 'zyx'
            })

            bvh.save('decompressor_Ytil.bvh', {
                'rotations': np.degrees(to_euler_np(Yrot)),
                'positions': 100.0 * Ypos,
                'offsets': 100.0 * Ypos[0],
                'parents': parents,
                'names': ['joint_%i' % i for i in range(utils.N_BONES)],
                'order': 'zyx'
            })
        except IOError as e:
            print(e)

        # # Write features
        #
        # fmin, fmax = Xgnd.cpu().numpy().min(), Xgnd.cpu().numpy().max()
        #
        # fig, axs = plt.subplots(nfeatures, sharex=True, figsize=(12, 2 * nfeatures))
        # for i in range(nfeatures):
        #     axs[i].plot(Xgnd[0, :500, i].cpu().numpy())
        #     axs[i].set_ylim(fmin, fmax)
        # plt.tight_layout()
        #
        # try:
        #     plt.savefig('decompressor_X.png')
        # except IOError as e:
        #     print(e)
        #
        # plt.close()
        #
        # # Write latent
        #
        # lmin, lmax = Zgnd.cpu().numpy().min(), Zgnd.cpu().numpy().max()
        #
        # fig, axs = plt.subplots(nlatent, sharex=True, figsize=(12, 2 * nlatent))
        # for i in range(nlatent):
        #     axs[i].plot(Zgnd[0, :500, i].cpu().numpy())
        #     axs[i].set_ylim(lmin, lmax)
        # plt.tight_layout()
        #
        # try:
        #     plt.savefig('decompressor_Z.png')
        # except IOError as e:
        #     print(e)
        #
        # plt.close()


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
    ax.set_aspect('equal')
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
                lines[i-1][0].set_xdata([positions_world[i, x], positions_world[skeleton_parents[i], x]])
                lines[i-1][0].set_ydata([positions_world[i, y], positions_world[skeleton_parents[i], y]])
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
    original_path = "decompressor_Ygnd.bvh"
    decompressed_path = "decompressor_Ytil.bvh"

    vis_skeleton(original_path, "original")
    vis_skeleton(decompressed_path, "reconstructed")
