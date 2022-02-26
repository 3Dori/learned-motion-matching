import torch.nn


class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        new_h = torch.tanh(self.linear(x))
        return new_h


mycell = torch.jit.script(MyCell())
mycell.save('mycell.pt')
