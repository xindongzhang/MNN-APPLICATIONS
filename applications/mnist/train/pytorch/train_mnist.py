import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim


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

        self.dense2 = nn.Linear(in_features=400, out_features=120, bias=False)
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
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
    validset = torchvision.datasets.MNIST(root='./data/', train=False, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=4, shuffle=False, num_workers=2)

    net  = MNIST()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(0, 2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("finish training")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))