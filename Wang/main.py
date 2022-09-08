import torch
from torch import nn, optim, device
import torch.nn.functional as F
import pandas as pd


pd = pd.read_csv('dataset.csv')


class Scheduler(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 4096)
        # use dropout and p = 0.1
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        return x


def train():
    model = Scheduler()
    # use Adam optimizer and learning rate is 0.002
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    epochs = 12

    print('Start')
    for epochs in range(epochs):
        running_loss = 0
        for data, ORFdata in pd:
            optimizer.zero_grad()
            # forward
            output = model(data)
            # compute loss
            loss = nn.CrossEntropyLoss(output, ORFdata)
            # backward
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            # close dropout when testing
            model.eval()
            for data, ORFdata in pd:
                data, target = data.to(device), ORFdata.to(device)
                # forecast
                log_ps = model(data)
                test_loss += nn.CrossEntropyLoss(log_ps, ORFdata, reduction='sum').item()
                prediction = ORFdata
                accuracy += prediction.eq(target.view_as(prediction)).sum().item()
            # open dropout
            model.train()

    print("times of learning: {}/{}.. ".format(epochs + 1, epochs),
          "loss of training: {:.3f}.. ".format(running_loss / len(pd)),
          "loss of testing: {:.3f}.. ".format(test_loss / len(pd)),
          "accuracy: {:.3f}".format(accuracy / len(pd)))

