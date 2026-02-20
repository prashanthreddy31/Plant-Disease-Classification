import timm
import torch.nn as nn

def get_deit(num_classes):
    model = timm.create_model('deit_base_patch16_224', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
