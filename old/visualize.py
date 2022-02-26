import mocap.datasets.cmu as CMU
from mocap.visualization.sequence import SequenceVisualizer


def visualize(seq):
    vis_dir = '/mnt/c/Users/nengn/cmu_vis'
    vis_name = 'CMU_VIS'

    vis = SequenceVisualizer(vis_dir, vis_name,  # mandatory parameters
                             plot_fn=None,  # TODO
                             vmin=-1, vmax=1,  # min and max values of the 3D plot scene
                             to_file=False,  # if True writes files to the given directory
                             subsampling=1,  # subsampling of sequences
                             with_pauses=False,  # if True pauses after each frame
                             fps=20,  # fps for visualization
                             mark_origin=False)  # if True draw cross at origin

    # plot single sequence
    vis.plot(seq,
             seq2=None,
             parallel=False,
             plot_fn1=None, plot_fn2=None,  # defines how seq/seq2 are drawn
             views=[(45, 45)],  # [(elevation, azimuth)]  # defines the view(s)
             lcolor='#099487', rcolor='#F51836',
             lcolor2='#E1C200', rcolor2='#5FBF43',
             noaxis=False,  # if True draw person against white background
             noclear=False, # if True do not clear the scene for next frame
             toggle_color=False,  # if True toggle color after each frame
             plot_cbc=None,  # alternatve plot function: fn(ax{matplotlib}, seq{n_frames x dim}, frame:{int})
             last_frame=None,   # {int} define the last frame < len(seq)
             definite_cbc=None,   # fn(ax{matplotlib}, iii{int}|enueration, frame{int})
             name='',
             plot_jid=False,
             create_video=False,
             video_fps=25,
             if_video_keep_pngs=False)
