from src.config import config
from src.utils import set_seed, get_device
from src.data_loader import prepare_data, get_transforms
from src.models.deit_model import get_deit
from src.models.efficientnet_model import get_efficientnet
from src.models.cnn_model import CustomCNN
from src.train import train_model

set_seed(config.SEED)


# ------------------------------
# Select Model
# ------------------------------
MODEL_NAME = "cnn"   # cnn / efficientnet / deit

# ------------------------------
# Transforms (Model-Specific)
# ------------------------------
train_transform, valid_transform = get_transforms(MODEL_NAME)

# ------------------------------
# Data Loaders
# ------------------------------
train_loader, valid_loader, test_loader, class_names = prepare_data(
    directory_root=config.DATA_DIR,
    train_transform=train_transform,
    valid_test_transform=valid_transform,
    batch_size=config.BATCH_SIZE
)

device = get_device()

# ------------------------------
# Model Selection
# ------------------------------
if MODEL_NAME.lower() == "cnn":
    model = CustomCNN(len(class_names))

elif MODEL_NAME.lower() == "efficientnet":
    model = get_efficientnet(len(class_names))

elif MODEL_NAME.lower() == "deit":
    model = get_deit(len(class_names))

else:
    raise ValueError("Invalid MODEL_NAME")

# ------------------------------
# Train
# ------------------------------
history = train_model(
    model,
    train_loader,
    valid_loader,
    device,
    config
)