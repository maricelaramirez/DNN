import torch
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

#The above is importing data (from pytorchbasics_data)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        # fully connected 1st layer (no. of pixels in, no. of neurons in hidden layers)
        # generally, fc(layer no.) = nn.linear(no. of inputs, no. of outputs)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)




net = Net()
print(net)

X = torch.rand((28,28))
X = X.view(-1, 28*28)

output = net(X)
print(output)

#Training begins here

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 5 # no. of times pass through entire network

for epoch in range(EPOCHS):
    for data in trainset: # data is a batch of featuresets abd labels
        X,y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        cost = F.nll_loss(output, y)
        cost.backward()
        optimizer.step()
    print(cost)

correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X,y = data
        output = net(X.view(-1, 784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print("Accuracy: ", round(correct/total, 3))

# show digit, print prediction
import matplotlib.pyplot as plt
plt.imshow(X[3].view(28,28))
plt.show()

print(torch.argmax(net(X[3].view(-1,784))[0]))