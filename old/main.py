import torch

from trainer import get_data_loader, train_compressor


def save_model(model, path):
    model.eval()
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, path)


train_loader = get_data_loader(x_filename='/mnt/c/Users/nengn/Documents/motionkeys/motionkey_new_x.npy',
                               y_filename='/mnt/c/Users/nengn/Documents/motionkeys/motionkey_new_y.npy')
compressor, decompressor = train_compressor(train_loader)

save_model(compressor, '/mnt/c/Users/nengn/Documents/Unreal Projects/UE4_MotionMatching/UE_MotMatch/Resources/compressor.pt')
save_model(decompressor, '/mnt/c/Users/nengn/Documents/Unreal Projects/UE4_MotionMatching/UE_MotMatch/Resources/decompressor.pt')
