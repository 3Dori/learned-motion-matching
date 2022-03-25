import numpy as np


COMPRESSOR_PATH = 'models/compressor.mdl'
DECOMPRESSOR_PATH = 'models/decompressor.mdl'
STEPPER_PATH = 'models/stepper.mdl'
PROJECTOR_PATH = 'models/projector.mdl'


def extract_locomotion_from_y_feature_vector(y, batch_size, window):
    y_pos = y[:, :, 0:75].reshape((batch_size, window, 25, 3))
    y_txy = y[:, :, 75:225].reshape((batch_size, window, 25, 3, 2))
    y_vel = y[:, :, 225:300].reshape((batch_size, window, 25, 3))
    y_ang = y[:, :, 300:375].reshape((batch_size, window, 25, 3))
    y_rvel = y[:, :, 375:378].reshape((batch_size, window, 3))
    y_rang = y[:, :, 378:381].reshape((batch_size, window, 3))

    return y_pos, y_txy, y_vel, y_ang, y_rvel, y_rang


def all_sequences_of_dataset(dataset):
    sequences = []
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            sequences.append((subject, action))
    return np.array(sequences)
