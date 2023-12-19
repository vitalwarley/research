from pathlib import Path

import torch
from models.insightface.recognition.arcface_torch.backbones import get_model
from torch import nn
from torch.nn import functional as F


# Step 1: Local Expert Layer
class LocalExpert(nn.Module):
    def __init__(self, input_size, hidden_size, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        z1 = self.fc1(x)
        z1 = F.leaky_relu(z1, self.negative_slope)
        z2 = self.fc2(z1)  # + x  # Residual connection
        z2 = F.sigmoid(z2)
        return z1, z2  # refined feature vector, prob for the i-th local expert


# Step 2: Cascading the Experts
class CascadedLocalExperts(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [LocalExpert(input_size if not i else hidden_size, hidden_size) for i in range(num_experts)]
        )

    def forward(self, x):
        predictions = []
        for i, expert in enumerate(self.experts):
            x, z = expert(x)  # z is the probability
            predictions.append(z)
        return torch.cat(predictions, dim=1)


# Step 3: Kinship Comparator with Cascaded Experts
class KinshipComparator(nn.Module):
    def __init__(self, input_size, hidden_size, num_kinship_relations):
        super().__init__()
        self.cascaded_experts = CascadedLocalExperts(input_size, hidden_size, num_kinship_relations)

    def forward(self, features, y):
        predictions = self.cascaded_experts(features)
        selected_prediction = predictions * y
        return selected_prediction


# Step 4: Final Network that combines everything
class MTCFNet(nn.Module):
    def __init__(self, weights: str = ""):
        super().__init__()
        # Create the backbone
        self.backbone = get_model("r100", fp16=True)  # TODO: enable selection of backbone
        if weights:
            print("Loaded insightface model.")
            self.backbone.load_state_dict(torch.load(weights))

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Create the comparator
        self.comparator = KinshipComparator(1024, 192, 11)
        # Add dropout layer
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, image1, image2, y):
        f1 = self.backbone(image1)
        f2 = self.backbone(image2)
        # print dtypes
        # kinship_one_hot = one_hot_encode_kinship(kinship_relation)
        combined_features = torch.cat((f1, f2), dim=1)
        combined_features = self.dropout(combined_features)
        comparison_output = self.comparator(combined_features, y)
        return comparison_output


# Create a test prediction as script
if __name__ == "__main__":
    HERE = Path(__file__).parent
    # Create a dummy input
    image1 = torch.randn(1, 3, 112, 112)
    image2 = torch.randn(1, 3, 112, 112)
    # Make y a one-hot vector
    y = torch.zeros(1, 11)
    y[0, 5] = 1
    # Create the model
    weights = HERE / "ms1mv3_arcface_r100_fp16.pth"
    model = MTCFNet(weights=str(weights))
    model.eval()
    # Make a prediction
    with torch.no_grad():
        prediction = model(image1, image2, y)
    # Print the prediction
    print(prediction)
