import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# ~~~~~~~~~~~~~~~~~~ Test ~~~~~~~~

# train_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('./data/', train=True, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ])),
#   batch_size=1, shuffle=True)
# print(next(enumerate(train_loader)))
#sys.exit()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

DATA="/home/akshay/Code/ML/Digit-Recognition-CNN/data/"
MODEL="/home/akshay/Code/ML/Digit-Recognition-CNN/model/"


#hyperparameters
learning_rate = 0.13
epochs = 100
batch_size=64
tbatch_size=6
log_interval = 10
momentum = 0.1
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * 42000 for i in range(epochs + 1)]



class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = CNN()
model = model.cuda()
model = model.double()

def traind(m, epochs, X, y):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum)

    tx = torch.from_numpy(X).to(device)
    ty = torch.from_numpy(y).to(device)

    ds = torch.utils.data.TensorDataset(tx, ty)

    train_loader = torch.utils.data.DataLoader(dataset=ds,
                                               batch_size=batch_size,
                                               shuffle=False)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epochs, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epochs - 1) * len(train_loader.dataset)))
            torch.save(model.state_dict(), './model/model.pth')
            torch.save(optimizer.state_dict(), './model/optimizer.pth')


def testd(X):

    model.eval()
    test_loss = 0
    correct = 0

    tx = torch.from_numpy(X).to(device)
    ds = torch.utils.data.TensorDataset(tx)

    test_loader = torch.utils.data.DataLoader(dataset=ds,
                                              batch_size=tbatch_size,
                                              shuffle=False)
    f = open("./data/out.txt", "w+")
    f.write("ImageId,Label\n")
    count=1
    with torch.no_grad():
        for data in test_loader:
            output = model(data[0])
            #print(((data[0].numpy()*255)+255)/2)
            pred = output.data.max(1, keepdim=True)[1]

            # if count < 60:
            #     fig=""
            #     plt.figure()
            #     for i in range(6):
            #         plt.subplot(2, 3, i + 1)
            #         plt.tight_layout()
            #         plt.imshow(data[0][i][0], cmap='gray',
            #                    interpolation='none')
            #         plt.title("Prediction: {}".format(
            #             output.data.max(1, keepdim=True)[1][i].item()))
            #         plt.xticks([])
            #         plt.yticks([])
            #     plt.show()

            #f.write(str(pred))
            for k in pred:
                k = k.item()
                f.write(str(count) + ',' + str(k) + "\n")
                count+=1
    f.close()


def main():

    # initialize training data
    train = pd.read_csv('/home/akshay/Code/ML/Digit-Recognition-CNN/data/train.csv')  # 42000 x 785
    m = train.shape[0]
    train_X = train.iloc[:, 1:].values
    train_X = ((train_X*2)-255) * (1 / 255)  # 42000 x 784
    train_X = np.resize(train_X, (m, 1, 28, 28))  # 42000 x 28 x 28

    train_y = train.iloc[:, 0].values.reshape(m)  # 42000 x 1

    # initialize testing data
    test = pd.read_csv('/home/akshay/Code/ML/Digit-Recognition-CNN/data/test.csv')  # 28000 x 784
    np.set_printoptions(threshold=np.nan)


    #test_X = test.values # 28000 x 784
    test_X = ((test.values*2)-255) * (1 / 255)  # 28000 x 784
    #test_X = test_X[:1]
    n = test_X.shape[0]
    test_X = np.resize(test_X, (n, 1, 28, 28))

    traind(m, epochs, train_X, train_y)
    testd(test_X)

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig.show()


if __name__ == '__main__':
    main()
