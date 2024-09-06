import torch.nn as nn
import torch.nn.functional as F
from components import ResidualBlock

class AgeEncoder(nn.Module):
    def __init__(self, input_dim=128):
        super(AgeEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class AgeClassifier():
    def predict(self, x):
        #x = mivolo(x)
        #return onehot(x)
        pass


class Encoder(nn.Module):
    def __init__(self, input_channels=3, num_filters=128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters // 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters // 4, num_filters // 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters // 2, num_filters, kernel_size=3, stride=2, padding=1)
        
        self.residual_block1 = ResidualBlock(num_filters)
        self.residual_block2 = ResidualBlock(num_filters)
        self.residual_block3 = ResidualBlock(num_filters)
        self.residual_block4 = ResidualBlock(num_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)

        return x

class Decoder(nn.Module):
    def __init__(self, input_channels=128, output_channels=3):
        super(Decoder, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels // 2, input_channels // 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels // 4, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = F.relu(self.conv1(x))
        
        x = self.upsample2(x)
        x = F.relu(self.conv2(x))
        
        x = self.conv3(x)
        x = F.sigmoid(x)  # Assuming the output image should be in the range [0, 1]
        
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.age_encoder = AgeEncoder()
        self.decoder = Decoder()

    def forward(self, x, age_vector):
        encoded_features = self.encoder(x)

        # Encode the age vector
        encoded_age_vector = self.age_encoder(age_vector).view(-1, 128, 1, 1)
        
        # Reweight the encoded feature maps with the encoded age vector
        reweighted_features = encoded_features * encoded_age_vector
        
        # Decode the reweighted feature maps to generate the output image
        output_image = self.decoder(reweighted_features)
        
        return output_image
    
class Discriminator(nn.Module):
    def __init__(self, input_channels=3, num_filters=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters // 8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_filters // 8, num_filters // 4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters // 4)
        self.conv3 = nn.Conv2d(num_filters // 4, num_filters // 2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters // 2)
        self.conv4 = nn.Conv2d(num_filters // 2, num_filters, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(num_filters)
        self.conv5 = nn.Conv2d(num_filters, num_filters, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(num_filters)
        self.conv6 = nn.Conv2d(num_filters, 1, kernel_size=4, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv6(x)
        x = F.sigmoid(x)  # Assuming the output should be in the range [0, 1]
        
        return x
