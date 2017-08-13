import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoduleNet(nn.Module):
    def __init__(self):
        super(NoduleNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv3d(1,32,3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32,32,3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(),
            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Dropout3d(),
            nn.Conv3d(64, 128, 3),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2))
        )
        self.classifer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 2),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal(m.weight)
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal(m.weight)

    def forward(self,x):
        x = self.feature(x)
        x = x.view(x.size(0),128)
        x = self.classifer(x)
        return x

class Flip(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample, z, x, y):
        cube, label = sample['cube'], sample['label']
        if (z == 1):
            cube = np.flip(cube, 0)
        if (x == 1):
            cube = np.flip(cube, 1)
        if (y == 1):
            cube = np.flip(cube, 2)

        return {'cube':cube, 'label':label}

class Crop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (20,output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample, z, x, y):
        cube, label = sample['cube'], sample['label']
        cube = cube[z:z+20, x:x+36, y:y+36]
        return {'cube':cube, 'label':label}

def test_data(model, cube):
    Fl = Flip((20, 36, 36))
    Cr = Crop((20, 36, 36))
    pred = []
    for cz in range(5):
        for cx in range(5):
            for cy in range(5):
                for fx in range(2):
                    for fy in range(2):
                        for fz in range(2):
                            tcube = model(cube)
                            tcube = tcube[0][0]
                            tcube = Fl(tcube, fz, fx, fy)
                            tcube = Cr(tcube, cz, cx, cy)
                            single_predict = model(tcube)
                            pred = pred + [single_predict[1]]
    return pred

if __name__ == '__main__':
    net = NoduleNet()

    input = Variable(torch.randn(1,1,36,36,20))
    output = net(input)

    net.zero_grad()
    output.backward(torch.randn(1,2))
    target = [] # dummy target
    criterion = nn.CrossEntropyLoss()
    print(output.size())
    print(target.size())
    loss = criterion(output, target)
    print(loss)
