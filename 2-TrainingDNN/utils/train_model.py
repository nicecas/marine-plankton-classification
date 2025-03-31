import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, ops
from torch.utils.data import Dataset
from torchvision.transforms import v2
import time
import os
import numpy as np
import copy
import csv
from collections import OrderedDict


def train_model(log_path, model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    # Compute dataset sizes for 'train' and 'val' phases
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    model = model.cuda()
    since = time.time()

    best_acc = 0.0
    best_epoch = 0
    best_model = None

    # Initialize early stopping parameters
    patience = 50  # Stop training if the validation loss does not decrease for 50 consecutive epochs
    max_patience = 50
    early_stop = False
    val_loss_min = np.inf  # Initialize the minimum validation loss as infinity

    # Remove the CSV file if it already exists
    if os.path.exists(log_path):
        os.remove(log_path)

    # Write CSV header if the log file doesn't exist yet
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "phase", "loss", "acc"])

    for epoch in range(num_epochs):
        print('-' * 10)
        print(f'Epoch {epoch}/{num_epochs - 1}')

        # Process both training and validation phases for each epoch
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data for the current phase
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Update running loss and correct predictions count
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            # Calculate loss and accuracy for the epoch-phase
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Save best model if validation accuracy improves
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch

            # Handle early stopping logic in validation phase
            if phase == 'val':
                val_loss_current = epoch_loss
                if val_loss_current < val_loss_min:
                    val_loss_min = val_loss_current
                    best_model = copy.deepcopy(model)
                    patience = max_patience  # Reset patience if validation loss decreases
                else:
                    patience -= 1

                if patience == 0:
                    early_stop = True
                    print(f'Early stopping at epoch {epoch}')
                    # Write the log entry for the current phase before breaking
                    with open(log_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, phase, f"{epoch_loss:.4f}", f"{epoch_acc:.4f}"])
                    break

            # Construct the log string for the current epoch and phase
            epoch_txt = f'Epoch {epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}'
            print(epoch_txt)

            # Write the log entry to the CSV file
            with open(log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, phase, f"{epoch_loss:.4f}", f"{epoch_acc:.4f}"])

        if early_stop:
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    best_acc_txt = f'Best val Acc: {best_acc:.4f} in Epoch {best_epoch}'
    print(best_acc_txt)

    return best_model, best_acc