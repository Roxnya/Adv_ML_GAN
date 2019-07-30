import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 2)
        self.hidden_f = torch.nn.LeakyReLU(0.2, inplace=True)

    def forward(self, noise):
        x = self.hidden_f(self.fc1(noise))
        x = self.hidden_f(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, 10)
        # prob value - greater value means more chance that it's drawn from the distribution
        self.fc3 = nn.Linear(10, 1)
        self.hidden_f = torch.nn.LeakyReLU(0.2, inplace=True)

    def forward(self, point):
        x = self.hidden_f(self.fc1(point))
        x = self.hidden_f(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
