import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, n, dropout_rate=0.05):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(n, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, n // 2)
        self.bn1 = nn.BatchNorm1d(n)
        self.bn2 = nn.BatchNorm1d(n)
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
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
    def __init__(self, n, kernel_size, stride=1, padding=1, dropout_rate=0.05):
        super(ConvolutionBlock, self).__init__()
        self.conv1 = nn.Conv1d(n, n, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(n, 1, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(n)
        self.bn2 = nn.BatchNorm1d(1)
        self.drop = nn.Dropout(dropout_rate)
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

if __name__ == "__main__":
    from base import load_pretrained_model
    from torch.utils.data import DataLoader
    from dataset import ageFIW
    import yaml
    import torch

    adaface, adaface_transform = load_pretrained_model()
    adaface.eval()

    config = yaml.safe_load(open("../params/rbkin.yml"))

    train_dataset = ageFIW(config['data_path'], "train_sort.csv", transform=adaface_transform, training=False)
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4, pin_memory=False)

    conv = ConvolutionBlock(5, kernel_size=3)

    images1, images2, labels = next(iter(train_loader))

    features1 = [adaface(img)[0] for img in images1]
    features2 = [adaface(img)[0] for img in images2]
    inputs1 = torch.stack(features1) # (batch_size, 5, 512)
    inputs2 = torch.stack(features2) # (batch_size, 5, 512)

    outputs1 = conv(inputs1) # (batch_size, 1, 512)
    outputs2 = conv(inputs2) # (batch_size, 1, 512)

    features = torch.cat([outputs1, outputs2], dim=1) # (batch_size, 2, 512)

    conv2 = ConvolutionBlock(2, kernel_size=3)
    features_conv2 = conv2(features) # (batch_size, 1, 512)

    res = ResidualBlock(512)
    features_conv2 = features_conv2.squeeze(1) # (batch_size, 512)

    breakpoint()