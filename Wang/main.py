import torch
from torch import nn, optim
import torch.nn.functional as F


class Scheduler(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 4096)
        # use dropout and p = 0.2
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        return x

def train():
    model = Scheduler()
    # use cross entropy loss function
    loss = nn.CrossEntropyLoss()
    # use Adam optimizer and learning rate is 0.002
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    epochs = 12
    # plot and compare
    train_losses, train_losses = [], []

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
                log_ps = model(data)
                test_loss += loss(log_ps, ORF)
                ps = torch.exp(log_ps)


    model.train()
    train_losses.append(running_loss / len(trainloader))
    test_losses.append(test_loss / len(testloader)