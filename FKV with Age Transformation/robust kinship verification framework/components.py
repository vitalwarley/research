import torch as nn

class ResidualBlock(nn.Module):
    def __init__(self, n):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(n, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, n/2)
        self.bn1 = nn.BatchNorm1d(n) # 1d or 2d?
        self.bn1 = nn.BatchNorm1d(n) # 1d or 2d?
        self.drop1 = nn.Dropout(0.05)
        self.drop2 = nn.Dropout(0.05)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        out = self.bn2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out
    
class ConvolutionBlock(nn.Module):
    def __init__(self, n, kernel_size, stride, padding):
        super(ConvolutionBlock, self).__init__()
        self.conv1 = nn.Conv1d(n, n, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(n, 1, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(n)
        self.bn2 = nn.BatchNorm1d(n)
        self.drop = nn.Dropout(0.05)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        return x