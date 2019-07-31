import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST(nn.Module):

    def __init__(self):
        super(MNIST, self).__init__()
        self.conv0  = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, bias=False)
        self.bn0    = nn.BatchNorm2d(num_features=6)
        self.relu0  = nn.ReLU()
        self.maxp0  = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv1  = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, bias=False)
        self.bn1    = nn.BatchNorm2d(num_features=16)
        self.relu1  = nn.ReLU()
        self.maxp1  = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.dense2 = nn.Linear(in_features=16 * 6 * 6, out_features=120, bias=False)
        self.dp2    = nn.Dropout(p=0.5)
        self.relu2  = nn.ReLU()
        self.dense3 = nn.Linear(in_features=120, out_features=10, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxp0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxp1(x)

        x = x.view(-1, self.num_flat_features(x))

        x = self.dense2(x)
        x = self.dp2(x)
        x = self.relu2(x)

        x = self.dense3(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    net  = MNIST()
    data = torch.randn(2, 1, 32, 32)
    out  = net(data)
    print(out)