import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import model
from tqdm import trange
from tqdm import tqdm


def train(train_loader, val_loader, block, epochs):
    """Function to define the model and train it"""
    eff_model = model.Model(block)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        eff_model.parameters(),
        lr=0.1,
        momentum=0.5
    )
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eff_model.to(device)
    train_loss_ls, val_loss_ls = [], []
    train_acc_ls, val_acc_ls = [], []

    for epoch in (pbar := trange(epochs)):
        eff_model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for img, label in tqdm(train_loader):
            img, label = img.to(device), label.to(device)

            # forward
            outputs = eff_model(img)
            loss = criterion(outputs, label)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del img, label, outputs, loss
            torch.cuda.empty_cache()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        train_loss_ls.append(avg_loss)

        train_acc = 100 * correct / total
        train_acc_ls.append(train_acc)

        eff_model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)

                outputs = eff_model(img)
                loss = criterion(outputs, label)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                del img, label, outputs, loss
                torch.cuda.empty_cache()

        val_loss_ls.append(val_loss / len(val_loader))
        val_acc = 100 * correct / total
        val_acc_ls.append(val_acc)

        pbar.set_description(
            f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_loss:.3f} - Val Loss: {val_loss_ls[-1]:.3f} - Train ACC: {train_acc:.3f} - Val ACC: {val_acc:.3f}")

    history = {"Train Loss": train_loss_ls, "Val Loss": val_loss_ls, 'Train Accuracy': train_acc_ls,
               'Val Accuracy': val_acc_ls}

    return eff_model, history
