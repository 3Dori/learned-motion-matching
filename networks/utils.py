import numpy as np


def extract_locomotion_from_y_feature_vector(y, batch_size, window):
    Ypos = y[:, :, 0:75].reshape((batch_size, window, 25, 3))
    Ytxy = y[:, :, 75:225].reshape((batch_size, window, 25, 3, 2))
    Yvel = y[:, :, 225:300].reshape((batch_size, window, 25, 3))
    Yang = y[:, :, 300:375].reshape((batch_size, window, 25, 3))
    Yrvel = y[:, :, 375:378].reshape((batch_size, window, 3))
    Yrang = y[:, :, 378:381].reshape((batch_size, window, 3))

    return Ypos, Ytxy, Yvel, Yang, Yrvel, Yrang


def randomly_pick_from_dataset(dataset, sequences, batch_size):
    probs = []
    for i, (subject, action) in enumerate(sequences):
        probs.append(dataset[subject][action]['n_frames'])
    probs = np.array(probs) / np.sum(probs)

    while True:
        idxs = np.random.choice(len(sequences), size=batch_size, replace=True, p=probs)
        # randomly pick batch_size pairs of frames
        for i, (subject, action) in enumerate(sequences[idxs]):
            action = dataset[subject][action]
            yield i, action
