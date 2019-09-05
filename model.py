import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.max_pool2d_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2d_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_1 = torch.nn.Linear(7 * 7 * 64, 128)
        self.linear_2 = torch.nn.Linear(128, 10)
        self.dropout2d = torch.nn.Dropout2d(p=0.5)
        self.dropout1d = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.max_pool2d_1(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.max_pool2d_2(x)
        x = x.reshape(x.size(0), -1)
        # x = self.dropout2d(x)

        # MLP
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout1d(x)
        pred = self.linear_2(x)

        return pred