import os
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from ptflops import get_model_complexity_info  # pip install ptflops

from MyDataLoader import MyNoiseDataset
from ONED_CNN_4 import OneD_CNN

BATCH_SIZE = 500
EPOCHS = 30
LEARNING_RATE = 0.0002


def init_weights(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight.data)


def create_data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size, shuffle=True)


def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    train_loss = 0
    model.train()
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = model(input)
        loss = loss_fn(prediction, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(data_loader)


def validate_single_epoch(model, data_loader, loss_fn, device):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            prediction = model(input)
            loss = loss_fn(prediction, target)
            val_loss += loss.item()
    return val_loss / len(data_loader)


def train(model, train_loader, val_loader, epochs, device, model_path, loss_path):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, val_losses = [], []
    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} | Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        train_loss = train_single_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate_single_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        print(f"Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print("----------------------------------")

    np.save(loss_path, {'train_losses': train_losses, 'val_losses': val_losses})
    print("Losses saved to", loss_path)
    print("Finished training")


def plot_learning_curve(loss_path, save_path='learning_curve.pdf'):
    data = np.load(loss_path, allow_pickle=True).item()
    train_losses, val_losses = data['train_losses'], data['val_losses']

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', marker='o', linestyle='--', color='#d62728', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', marker='s', linestyle='-.', color='#2ca02c', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.title("Learning Curve", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()
    print("Learning curve saved to", save_path)


def Train_Validate_CNN(train_file, val_file, model_path, loss_path, file_sheet):
    train_data = MyNoiseDataset(train_file, file_sheet)
    val_data = MyNoiseDataset(val_file, file_sheet)
    train_loader = create_data_loader(train_data, BATCH_SIZE)
    val_loader = create_data_loader(val_data, BATCH_SIZE)

    model = OneD_CNN()
    model.apply(init_weights)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params} ({total_params/1e3:.2f} K)")

    # Compute MACs per frame
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (1, 3000), as_strings=True, verbose=True)
        print(f"MACs per frame: {macs}")
        print(f"Parameters: {params}")

    train(model, train_loader, val_loader, EPOCHS, device, model_path, loss_path)
    plot_learning_curve(loss_path)


# Dataset paths and model saving paths
train_file = 'Dataset_v2/Synthesized_Dataset_3khz/Training_data'
val_file = 'Dataset_v2/Synthesized_Dataset_3khz/Validate_data'
MODEL_PTH = 'models/Pretrained_CNN_v2.pth'
LOSS_PTH = 'models/losses.npy'
File_sheet = 'Soft_Index.csv'

Train_Validate_CNN(train_file, val_file, MODEL_PTH, LOSS_PTH, File_sheet)
