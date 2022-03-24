import numpy as np


COMPRESSOR_PATH = 'models/compressor.mdl'
DECOMPRESSOR_PATH = 'models/decompressor.mdl'
STEPPER_PATH = 'models/stepper.mdl'


def extract_locomotion_from_y_feature_vector(y, batch_size, window):
    Ypos = y[:, :, 0:75].reshape((batch_size, window, 25, 3))
    Ytxy = y[:, :, 75:225].reshape((batch_size, window, 25, 3, 2))
    Yvel = y[:, :, 225:300].reshape((batch_size, window, 25, 3))
    Yang = y[:, :, 300:375].reshape((batch_size, window, 25, 3))
    Yrvel = y[:, :, 375:378].reshape((batch_size, window, 3))
    Yrang = y[:, :, 378:381].reshape((batch_size, window, 3))

    return Ypos, Ytxy, Yvel, Yang, Yrvel, Yrang


def all_sequences_of_dataset(dataset):
    sequences = []
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            sequences.append((subject, action))
    return np.array(sequences)
