import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, filepath):
        set_T = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        set = set_T.T
        self.len = set.shape[0]
        self.data_X = torch.from_numpy(set[:, :12])
        self.data_Y = torch.from_numpy(set[:, 12:])

    def __getitem__(self, index):
        return self.data_X[index], self.data_Y[index]

    def __len__(self):
        return self.len


file = "./datasetZ.csv"
file_test = "./testsetZ.csv"

dataset = MyDataset(file)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=2)
testset = MyDataset(file_test)
testloader = DataLoader(dataset=testset, batch_size=64, shuffle=True, num_workers=2)


class Net(nn.Module):
    def __init__(self, ngpu):
        super(Net, self).__init__()
        self.ngpu = ngpu
        self.fc1 = nn.Linear(12, 144)
        self.fc2 = nn.Linear(144, 256)
        self.fc3 = nn.Linear(256, 66)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self(F.relu(self.fc1(x)))
        x = self(F.relu(self.fc2(x)))
        x = self(F.relu(self.fc3(x)))

        # x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        # x = self.dropout(F.relu(self.fc3(x)))

        return x


epochs = 200
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
N = Net(ngpu).to(device)
criterion = nn.MSELoss
optimizer = torch.optim.SGD(N.parameters(), lr=0.002)


def train():
    for j in range(0, epochs):
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            output = Net(data[0])
            Y = data[1]
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()

            if (i + 1) % 2 == 0:
                print('Epoch_number [{}/50000/{}/{}],loss: {:.6f},'.format(
                    64 * i, j + 1, epochs, loss.item(), ))


if __name__ == "__main__":
    train()

# def test():
#     running_loss=0
#     model = Scheduler()
#     test_loss = 0
#     accuracy = 0
#     epochs = 15
#     with torch.no_grad():
#         model.eval()
#     for i, data in enumerate(testloader, 0):
#         Z = data[1]
#         print(Z)
#         output = Scheduler(Z)
#         output = output
#         print(output)
#
#
#         # X, Y = X.to(device), Y.to(device)
#         # log_ps = model(X)
#         # test_loss += nn.CrossEntropyLoss(log_ps, Y).item()
#         # prediction = Y
#         # accuracy += prediction.eq(Y.view_as(prediction)).sum().item()
#         # model.train()
#
#     # print("times of learning: {}/{}.. ".format(epochs + 1, epochs),
#     #       "loss of training: {:.3f}.. ".format(running_loss / len(dataloader)),
#     #       "loss of testing: {:.3f}.. ".format(test_loss / len(dataloader)),
#     #       "accuracy: {:.3f}".format(accuracy / len(dataloader)))
