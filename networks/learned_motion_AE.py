import torch
from torch import nn
import torch.nn.functional as F


class Compressor(nn.Module):
    def __init__(self, dataset, input_size, output_size, hidden_size=512):
        super().__init__()

        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        device = dataset.device()
        y_mean = torch.as_tensor(dataset.Y_mean, dtype=torch.float32, device=device)
        q_mean = torch.as_tensor(dataset.Q_mean, dtype=torch.float32, device=device)
        self.compressor_mean = torch.cat([y_mean, q_mean], dim=-1)

        y_scale = torch.as_tensor(dataset.Y_compressor_scale, dtype=torch.float32, device=device)
        q_scale = torch.as_tensor(dataset.Q_compressor_scale, dtype=torch.float32, device=device)
        self.compressor_scale = torch.cat([y_scale, q_scale], dim=-1)

    def forward(self, y):
        n_batch, n_window = y.shape[:2]
        y = y.reshape((n_batch * n_window, -1))
        y = F.elu(self.linear0(y))
        y = F.elu(self.linear1(y))
        y = F.elu(self.linear2(y))
        y = self.linear3(y)
        return y.reshape((n_batch, n_window, -1))

    def compress(self, y, q):
        return self((torch.cat([y, q], dim=-1) - self.compressor_mean) / self.compressor_scale)


class Decompressor(nn.Module):
    def __init__(self, dataset, input_size, output_size, hidden_size=512):
        super().__init__()

        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, output_size)

        device = dataset.device()
        self.y_scale = torch.as_tensor(dataset.Y_decompressor_scale, dtype=torch.float32, device=device)
        self.y_mean = torch.as_tensor(dataset.Y_mean, dtype=torch.float32, device=device)

    def forward(self, x):
        n_batch, n_window = x.shape[:2]
        x = x.reshape((n_batch * n_window, -1))
        x = F.relu(self.linear0(x))
        x = self.linear1(x)
        return x.reshape((n_batch, n_window, -1))

    def decompress(self, x, z=None):
        x_z = x if z is None else torch.cat([x, z], dim=-1)
        return self(x_z) * self.y_scale + self.y_mean


class Stepper(nn.Module):
    def __init__(self, dataset, param_size, hidden_size=512):
        super().__init__()

        self.linear0 = nn.Linear(param_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, param_size)

        device = dataset.device()
        self.stepper_mean_in = torch.as_tensor(dataset.stepper_mean_in, dtype=torch.float32, device=device)
        self.stepper_std_in = torch.as_tensor(dataset.stepper_std_in, dtype=torch.float32, device=device)
        self.stepper_mean_out = torch.as_tensor(dataset.stepper_mean_out, dtype=torch.float32, device=device)
        self.stepper_std_out = torch.as_tensor(dataset.stepper_std_out, dtype=torch.float32, device=device)
        self.fps = dataset.fps()

    def forward(self, x):
        n_batch, n_window = x.shape[:2]
        x = x.reshape((n_batch * n_window, -1))
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x.reshape((n_batch, n_window, -1))

    def predict_x_z(self, x_z, window):
        predicted_x_z = [x_z[:, 0:1]]
        for _ in range(window - 1):
            delta_x_z = (self((predicted_x_z[-1] - self.stepper_mean_in) / self.stepper_std_in)
                         * self.stepper_std_out + self.stepper_mean_out)
            predicted_x_z.append(predicted_x_z[-1] + delta_x_z / self.fps)
        return torch.cat(predicted_x_z, dim=1)


class Projector(nn.Module):
    def __init__(self, dataset, input_size, output_size, hidden_size=512):
        super().__init__()

        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

        device = dataset.device()
        self.projector_mean_in = torch.as_tensor(dataset.projector_mean_in, dtype=torch.float32, device=device)
        self.projector_std_in = torch.as_tensor(dataset.projector_std_in, dtype=torch.float32, device=device)
        self.projector_mean_out = torch.as_tensor(dataset.projector_mean_out, dtype=torch.float32, device=device)
        self.projector_std_out = torch.as_tensor(dataset.projector_std_out, dtype=torch.float32, device=device)

    def forward(self, x):
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)

    def project(self, x_hat):
        return (self((x_hat - self.projector_mean_in) / self.projector_std_in)
                * self.projector_std_out + self.projector_mean_out)
