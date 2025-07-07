import torch
from torchvision import models
import torch.nn as nn
from tqdm import trange
from tqdm import tqdm


class Model:
    def __init__(self, out_features) -> None:
        """Function to initialize, load the pretrained model then change the output layers in the required format"""
        self.model = models.efficientnet_b0(pretrained=True)
        self.in_features = self.model.classifier[1].in_features
        self.out_features = out_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True),
            nn.ReLU()
        )

    def parameters(self):
        """Function to return the model parameters"""
        return self.model.parameters()

    def state_dict(self):
        """Function to return the state_dict of the model"""
        return self.model.state_dict()

    def loss(self, out1, out2, y, m=1.0):
        """Function to Calculate the loss"""
        dist = torch.norm(out1 - out2, p=2, dim=1)
        loss1 = y * (dist ** 2)
        loss2 = (1 - y) * torch.clamp(m ** 2 - dist ** 2, min=0)

        return torch.mean(loss1 + loss2)

    def train(self, train_loader, val_loader, epochs, optimizer, scheduler, m=1.0):
        """Function to train the model"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        train_loss_ls, val_loss_ls = [], []

        for epoch in (pbar := trange(epochs)):
            self.model.train()
            running_loss = 0.0

            for img1, img2, label in tqdm(train_loader):
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                optimizer.zero_grad()

                # forward
                outputs1 = self.model(img1)
                outputs2 = self.model(img2)
                loss = self.loss(outputs1, outputs2, label, m)
                running_loss += loss.item()
                # backward
                loss.backward()
                optimizer.step()
                del img1, img2, label, outputs1, outputs2, loss
                torch.cuda.empty_cache()

            scheduler.step()
            avg_loss = running_loss / len(train_loader)
            train_loss_ls.append(avg_loss)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for img1, img2, label in val_loader:
                    img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                    optimizer.zero_grad()

                    outputs1 = self.model(img1)
                    outputs2 = self.model(img2)
                    loss = self.loss(outputs1, outputs2, label)
                    val_loss += loss.item()

                    del img1, img2, label, outputs1, outputs2, loss
                    torch.cuda.empty_cache()

            val_loss_ls.append(val_loss/len(val_loader))
            pbar.set_description(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_loss:.3f} - Val Loss: {val_loss_ls[-1]:.3f}")

        return train_loss_ls, val_loss_ls

    def eval(self, val_loader, m=1.0):
        """Function to evaluate the model"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                outputs1 = self.model(img1)
                outputs2 = self.model(img2)
                loss = self.loss(outputs1, outputs2, label, m)
                val_loss += loss.item()

                del img1, img2, label, outputs1, outputs2, loss
                torch.cuda.empty_cache()

        return val_loss / len(val_loader)
