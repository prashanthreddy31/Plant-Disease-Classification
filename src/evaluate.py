import torch
from sklearn.metrics import accuracy_score

def evaluate(model, loader, device):

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    return accuracy_score(y_true, y_pred)