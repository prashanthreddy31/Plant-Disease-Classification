import timm
import torch.nn as nn

def get_efficientnet(num_classes):
    model = timm.create_model('efficientnet_b0', pretrained=True)
    # Freezing the layers
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model