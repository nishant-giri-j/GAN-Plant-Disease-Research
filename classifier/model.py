import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # replace last layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
