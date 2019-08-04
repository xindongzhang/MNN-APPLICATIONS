import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim


class MNIST(nn.Module):

    def __init__(self):
        super(MNIST, self).__init__()
        self.conv0  = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, bias=False)
        self.bn0    = nn.BatchNorm2d(num_features=20)
        self.relu0  = nn.ReLU()
        self.maxp0  = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv1  = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, bias=False)
        self.bn1    = nn.BatchNorm2d(num_features=50)
        self.relu1  = nn.ReLU()
        self.maxp1  = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2  = nn.Conv2d(in_channels=50, out_channels=500, kernel_size=4, stride=1, bias=False)
        self.bn2    = nn.BatchNorm2d(num_features=500)
        self.relu2  = nn.ReLU()
        self.conv3  = nn.Conv2d(in_channels=500, out_channels=10, kernel_size=1, stride=1, bias=False)
        # self.dense2 = nn.Linear(in_features=400, out_features=120, bias=False)
        # self.dp2    = nn.Dropout(p=0.5)
        # self.relu2  = nn.ReLU()
        # self.dense3 = nn.Linear(in_features=120, out_features=10, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxp0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxp1(x)

        # x = x.view(-1, self.num_flat_features(x))
        # x = self.dense2(x)
        # x = self.dp2(x)
        # x = self.relu2(x)
        # x = self.dense3(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = torch.squeeze(x)
        return x


if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data/MNIST', train=True, transform=transform, download=False)
    validset = torchvision.datasets.MNIST(root='./data/MNIST', train=False, transform=transform, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=4, shuffle=False, num_workers=2)

    net  = MNIST()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(0, 10):
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
    torch.save(net.state_dict(), "./mnist.pth")
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