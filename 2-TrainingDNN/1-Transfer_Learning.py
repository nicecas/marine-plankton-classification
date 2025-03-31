import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision.transforms import v2
import os
from utils.choose_model import choose_model
from utils.train_model import train_model

if __name__ == '__main__':
    # Define common data augmentations
    data_transforms = {
        'train': v2.Compose([
            v2.PILToTensor(),
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            v2.ToDtype(torch.float32, scale=True),  # Convert to float32 and scale to [0, 1]
            v2.Normalize(mean=(0.704, 0.740, 0.781), std=(0.115, 0.135, 0.160))
        ]),
        'val': v2.Compose([
            v2.PILToTensor(),
            v2.Resize(256, antialias=True),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.704, 0.740, 0.781), std=(0.115, 0.135, 0.160))
        ]),
    }

    # Define the dataset path
    data_dir = r'../0-Data/2-blobs'

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32,
                                               shuffle=True, num_workers=8, pin_memory=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=1,
                                             shuffle=False, num_workers=8, pin_memory=True)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    model_names = [
        # 'resnet18', 'resnet50', 'resnet101',
        # 'efficientnetv2l', 'efficientnetv2m',
        'efficientnetv2s',
        'swinv2b', 'swinv2s', 'swinv2t'
    ]

    # Define optimizer configuration dictionary with lambda functions for optimizer creation
    optimizers_cfg = {
        "SGD": lambda params: torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0001),
        "Adam": lambda params: torch.optim.Adam(params, lr=0.001)
    }

    log = []

    # Iterate over each model
    for model_name in model_names:
        print(f"Training model: {model_name}")
        # Iterate over each optimizer
        for optimizer_name, opt_fn in optimizers_cfg.items():
            print(f"  Using optimizer: {optimizer_name}")

            # Create a result folder for all models
            result_folder = './models'
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)

            # Prepare log file
            log_filename = model_name + f'_{optimizer_name}_lr0.001.csv'
            log_path = os.path.join(result_folder, log_filename)
            if os.path.exists(log_path):
                os.remove(log_path)

            # Delete the previous model if it exists
            if 'model' in globals():
                del globals()['model']

            # Load the model
            model = choose_model(model_name, len(class_names))

            # Define loss function
            criterion = nn.CrossEntropyLoss()

            # Create optimizer using the current optimizer configuration
            optimizer_ft = opt_fn(model.parameters())

            # Define learning rate scheduler
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.5)

            # Train the model
            best_model, best_acc = train_model(log_path, model, dataloaders, criterion,
                                               optimizer_ft, exp_lr_scheduler, num_epochs=200)

            # Save the best model with the model name, optimizer, and learning rate in the filename
            torch.save(best_model, os.path.join(result_folder, f'{model_name}_{optimizer_name}_lr0.001_best.pt'))

            log.append([model_name, optimizer_name, best_acc])

    print("Training completed. Summary:")
    for record in log:
        print(record)