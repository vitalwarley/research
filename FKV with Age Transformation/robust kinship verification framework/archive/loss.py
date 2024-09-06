import torch as nn
import torch.nn.functional as F

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def generator_loss(self, D_fake):
        return self.mse_loss(D_fake, nn.ones_like(D_fake))

    def discriminator_loss(self, D_real, D_fake):
        real_loss = self.mse_loss(D_real, nn.ones_like(D_real))
        fake_loss = self.mse_loss(D_fake, nn.zeros_like(D_fake))
        return (real_loss + fake_loss) / 2

def age_classification_loss(predicted, target):
    pass

def reconstruction_loss(predicted, original):
    return nn.mean(nn.abs(predicted - original))

def identity_loss(x, synthesized_image, model): # model = adaface
    
    predicted_features = model(x)
    target_features = model(synthesized_image)
    return F.mse_loss(predicted_features, target_features)