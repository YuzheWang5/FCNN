import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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


file = "./mytrainset.csv"
# file_test = "./mytestset.csv"

dataset = MyDataset(file)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=2)


# testset = MyDataset(file_test)
# testloader = DataLoader(dataset=testset, batch_size=64, shuffle=True, num_workers=2)


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(12, 144),
            nn.ReLU(True),
            nn.Linear(144, 256),
            nn.ReLU(True),
            nn.Linear(256, 66),
        )

    def forward(self, x):
        x = self.dis(x)
        return x


model = NET()
epochs = 100
criterion = nn.CrossEntropyLoss()
optimizier = torch.optim.Adam(model.parameters(), lr=0.0003)


def train():
    for j in range(0, epochs):
        for i, data in enumerate(dataloader, 0):
            optimizier.zero_grad()
            Y = data[1]
            output = model(data[0])
            loss = criterion(output, Y)
            loss.backward()
            optimizier.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/10000/{}/{}],loss: {:.6f},'.format(
                    64 * i, j + 1, epochs, loss.item(), ))

            torch.save(model.state_dict(), r"C:\Users\woshi\FCNN\NN.pth")


# def test():
#     N.load_state_dict(torch.load(r'C:\Users\kakashisa\PycharmProjects\pythonProject\NN.pth'))
#     for i, data in enumerate(testloader1, 0):
#         Z = data[1].cuda()
#         print(Z)
#         output = N(Z).cuda()
#         output = output
#         print(output)

if __name__ == "__main__":
    train()
    # test()
