from __future__ import print_function  # only relevant for Python 2
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd, gluon, autograd

def data_xform(data):
    """Move channel axis to the beginning, cast to float32, and normalize to [0, 1]."""
    return nd.moveaxis(data, 2, 0).astype('float32') / 255

class MNIST(mx.gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(MNIST, self).__init__(**kwargs)
        self.conv0  = nn.Conv2D(channels=20, kernel_size=(5, 5), activation='relu')
        self.maxp0  = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.conv1  = nn.Conv2D(channels=50, kernel_size=(5, 5), activation='relu')
        self.maxp1  = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.flaten = nn.Flatten()
        self.dense0 = nn.Dense(units=500, activation='relu')
        self.dense1 = nn.Dense(units=10, activation=None)

    def hybrid_forward(self, F, x):
        x = self.conv0(x)
        x = self.maxp0(x)
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.flaten(x)
        x = self.dense0(x)
        x = self.dense1(x)
        return x


train_data = mx.gluon.data.vision.MNIST(root='./data', train=True).transform_first(data_xform)
valid_data = mx.gluon.data.vision.MNIST(root='./data', train=False).transform_first(data_xform)

print(train_data)

batch_size = 32
train_loader = mx.gluon.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = mx.gluon.data.DataLoader(valid_data, shuffle=False, batch_size=batch_size)

net = MNIST()

net.initialize(mx.init.Xavier())

trainer = mx.gluon.Trainer(
    params=net.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.04},
)

# training
num_epochs = 10
metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        with autograd.record():
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
        loss.backward()
        metric.update(labels, outputs)
        trainer.step(batch_size=inputs.shape[0])
    # Print the evaluation metric and reset it for the next epoch
    name, acc = metric.get()
    print('After epoch {}: {} = {}'.format(epoch + 1, name, acc)) 
    metric.reset()

# testing
metric = mx.metric.Accuracy()
for inputs, labels in valid_loader:
    # Possibly copy inputs and labels to the GPU
    metric.update(labels, net(inputs))
print('Validaton: {} = {}'.format(*metric.get()))

## export model
net.hybridize()
x = mx.nd.random_normal(shape=(1, 1, 28, 28))
net(x)
net.export('mnist')
