import torch.cuda

from common.locomotion_utils import compute_z_vector
from networks.train_stepper import StepperTrainer
from networks.utils import COMPRESSOR_PATH, DECOMPRESSOR_PATH, STEPPER_PATH
from common.dataset_locomotion import load_dataset


if __name__ == '__main__':
    dataset = load_dataset()
    # print('Start training decompressor')
    # decompressor_trainer = DecompressorTrainer()
    # compressor, decompressor = decompressor_trainer.train(dataset)
    # torch.save(compressor, 'models/compressor.mdl')
    # torch.save(decompressor, 'models/decompressor.mdl')
    compressor = torch.load(COMPRESSOR_PATH)
    decompressor = torch.load(DECOMPRESSOR_PATH)
    print('Building Z vector')
    compute_z_vector(dataset, compressor)
    print('Start training stepper')
    stepper_trainer = StepperTrainer(compressor)
    stepper = stepper_trainer.train(dataset)
    torch.save(stepper, STEPPER_PATH)
    print('Done')
