import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.early_stopping import EarlyStopping
from src.config import config

def train_model(model, train_loader, val_loader, device, config):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.3)

    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        path=config.MODEL_SAVE_PATH
    )

    model.to(device)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(config.EPOCHS):

        # TRAINING
        model.train()
        running_loss = 0
        running_corrects = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects+= torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects.double() / len(train_loader)
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        # VALIDATION
        model.eval()
        running_loss_val = 0.0
        running_corrects_val = 0   

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss_val += loss.item()
                _, preds = torch.max(outputs, 1)
                running_corrects_val += torch.sum(preds == labels.data)

        epoch_loss_val = running_loss_val / len(val_loader)
        epoch_acc_val  = running_corrects_val.double() / len(val_loader)
        history["val_loss"].append(epoch_loss_val)
        history["val_acc"].append(epoch_acc_val.item())


        # Scheduler step
        scheduler.step(epoch_loss_val)

        
        history["val_loss"].append(epoch_loss_val)

        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_loss_val:.4f}")

        # Early stopping
        early_stopping(epoch_loss_val, model)

        if early_stopping.early_stop:
            print("🛑 Early stopping triggered!")
            break

    return history


