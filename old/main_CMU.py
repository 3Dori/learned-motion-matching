import mocap.datasets.cmu as CMU

from visualize import visualize
from trainer import train


def encode_decode_seq(model, input_seq):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_seq = input_seq.copy()
    seq = torch.tensor(input_seq)
    input_shape = seq.shape[1] * seq.shape[2]    # the original shape is (# of joints, # of dimensions)
    for i, frame in enumerate(seq):
        output_frame = model(frame.view(-1, input_shape).to(device))
        output_seq[i] = torch.Tensor.cpu(output_frame).detach().numpy().reshape((seq.shape[1], seq.shape[2]))
    return output_seq


train_loader = get_data_loader(x_filename='/mnt/c/Users/nengn/Documents/motionkeys/motionkey_1645545076_x.npy',
                               y_filename='/mnt/c/Users/nengn/Documents/motionkeys/motionkey_1645545076_y.npy')
compressor, decompressor = train_compressor(train_loader)

# ds = CMU.CMU(subjects=['10'])
# seq_subject_10 = ds[0]
# seq_subject_10 = encode_decode_seq(AE_model, seq_subject_10)
# visualize(seq_subject_10)
