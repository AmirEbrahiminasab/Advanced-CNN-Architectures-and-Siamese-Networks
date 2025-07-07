import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import model


def train(out_features, train_loader, val_loader, step_size, gamma, learning_rate, epochs, m=1.0):
    """Function to define the model like part 1"""
    eff_model = model.Model(out_features=out_features)
    optimizer = optim.SGD(
        eff_model.parameters(),
        lr=learning_rate,
        momentum=0.9
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    train_loss, val_loss = eff_model.train(train_loader, val_loader, epochs, optimizer, scheduler, m)

    return eff_model, train_loss, val_loss
