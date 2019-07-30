import numpy as np
import torch.nn as nn
import torch

gen_hid_1 = 50
gen_hid_2 = 100
gen_hid_3 = 300
gen_out = 2
#num_classes = 2

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.04)

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, gen_hid_1)
        self.fc1.apply(init_weights)
        self.fc2 = nn.Linear(gen_hid_1, gen_hid_2)
        self.fc2.apply(init_weights)
        self.fc3 = nn.Linear(gen_hid_2, gen_hid_3)
        self.fc3.apply(init_weights)
        self.fc4 = nn.Linear(gen_hid_3, gen_out)
        self.fc4.apply(init_weights)
        self.drop1 = nn.Dropout(p=0.4)
        self.drop2 = nn.Dropout(p=0.4)
        self.drop3 = nn.Dropout(p=0.4)
        self.bn0 = nn.BatchNorm1d(input_size)
        self.bn1 = nn.BatchNorm1d(gen_hid_1)
        self.bn2 = nn.BatchNorm1d(gen_hid_2)
        self.hidden_f = torch.nn.LeakyReLU(0.01, inplace=True)

    def forward(self, noise):
        x = self.drop1(self.hidden_f(self.fc1(noise)))
        x = self.drop2(self.hidden_f(self.fc2(x)))
        x = self.drop3(self.hidden_f(self.fc3(x)))
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 50)
        # prob value - greater value means more chance that it's drawn from the distribution
        self.fc3 = nn.Linear(50, 1)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.3)
        self.hidden_f = torch.nn.LeakyReLU(0.01, inplace=True)

    def forward(self, point):
        x = self.hidden_f(self.fc1(point))
        x = self.hidden_f(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
