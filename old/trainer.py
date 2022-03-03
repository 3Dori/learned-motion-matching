from torch.utils.data import DataLoader

from torch import optim, nn
import torch.cuda

from autoencoder import Autoencoder
from networks.learned_motion_AE import Compressor, Decompressor
from data import CMUDataset, UE4DataSet


def get_CMU_data_loader(subjects, batch_size=128, shuffle=True, num_workers=2):
    dataset = CMUDataset(subjects)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_data_loader(x_filename, y_filename, batch_size=128, shuffle=True, num_workers=2):
    dataset = UE4DataSet(x_filename, y_filename)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train(train_loader, test_loader=None, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = train_loader.dataset.input_size
    model = Autoencoder(input_shape=input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        loss = 0
        test_loss = 0
        for batch_features, _ in train_loader:
            batch_features = batch_features.view(-1, input_size).to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            train_loss = criterion(outputs, batch_features)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        loss /= len(train_loader)
        print(f'Epoch : {epoch+1}/{epochs}, loss = {loss:.6f}')

    return model


def train_compressor(train_loader, test_loader=None, epochs=100):
    from itertools import chain
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_size = train_loader.dataset.x_size
    y_size = train_loader.dataset.y_size
    compressor = Compressor(n_layers=5, n_units=128, input_shape=y_size).to(device)
    decompressor = Decompressor(n_layers=3, n_units=128, input_shape=x_size, output_shape=y_size).to(device)
    optimizer = optim.Adam(chain(compressor.parameters(), decompressor.parameters()), lr=1e-3)
    criterion = nn.MSELoss()    # TODO: use the criterion in the paper

    for epoch in range(epochs):
        loss = 0
        test_loss = 0
        for (x, y), _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            z = compressor(y)
            reconstructed_y = decompressor(x, z)
            train_loss = criterion(y, reconstructed_y)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        loss /= len(train_loader)
        print(f'Epoch : {epoch+1}/{epochs}, loss = {loss:.6f}')

    return compressor, decompressor
