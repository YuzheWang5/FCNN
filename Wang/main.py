import torch
from torch import nn, optim, device
import torch.nn.functional as F


class Scheduler(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 4096)
        # use dropout to avoid overfitting and p = 0.1
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        return x

def train():
    global running_loss
    model = Scheduler()
    # use cross entropy loss function
    loss = nn.CrossEntropyLoss()
    # use Adam optimizer and learning rate is 0.002
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    epochs = 12

    print('Start')
    for epochs in range(epochs):
        running_loss = 0
        for data, ORF in trainloader:
            optimizer.zero_grad()
            log_ps = model(data)
            loss = loss(log_ps, ORF)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            # close dropout when testing
            model.eval()
            for data, ORF in testloader:
                data, target = data.to(device), ORF.to(device)
                test_loss += nn.CrossEntropyLoss(data, ORF, reduction='sum').item()  # sum up batch loss
                pred = ORF
                accuracy += pred.eq(target.view_as(pred)).sum().item()
            model.train()
            # storing the loss of training/testing in lists and plot

    print("times of learning: {}/{}.. ".format(epochs + 1, epochs),
          "loss of training: {:.3f}.. ".format(running_loss / len(trainloader)),
          "loss of testing: {:.3f}.. ".format(test_loss / len(testloader)),
          "accuracy: {:.3f}".format(accuracy / len(testloader)))
