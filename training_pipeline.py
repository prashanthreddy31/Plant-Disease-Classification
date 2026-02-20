from src.config import config
from src.utils import set_seed, get_device
from src.data_loader import prepare_data
from src.models.deit_model import get_deit
from src.models.efficientnet_model import get_efficientnet
from src.models.cnn_model import CustomCNN
from src.train import train_model

set_seed(config.SEED)

device = get_device()

train_loader, val_loader, test_loader, classes = prepare_data(
    config.DATA_DIR, config.Image_size, config.BATCH_SIZE
)

## Select any of the models(CustomCNN, Efficientnet, DeiT)
model = get_deit(len(classes))

history = train_model(model, train_loader, val_loader, device, config)