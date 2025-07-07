import numpy as np
import torch
from tqdm import tqdm


def pred(test_loader, model):
    """Function to return prediction of model based on test set"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    y_pred = []
    y_true = []
    correct, total = 0, 0
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img, label = img.to(device), label.to(device)

            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            y_pred = y_pred + predicted.tolist()
            y_true = y_true + label.tolist()
            total += label.size(0)
            correct += (predicted == label).sum().item()

            del img, label, outputs
            torch.cuda.empty_cache()

    print(100 * correct / total)
    return y_pred, y_true


def embedding(test_loader, model):
    """Function to capture embedding representation of the bottleneck of given model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, label in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            batch_embeddings = outputs.view(outputs.size(0), -1).cpu().numpy()
            embeddings.append(batch_embeddings)
            labels.append(label.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    return embeddings, labels


def layer_by_layer(img, model):
    """Function to capture each layer output for visualizing it later on."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    img = img.unsqueeze(0).to(device)

    activations = []

    def hook_fn(module, input, output):
        """Function to hook and capture output of each layer."""
        activations.append(output)

    hooks = []
    for name, layer in model.named_children():
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    with torch.no_grad():
        model(img)
    for hook in hooks:
        hook.remove()

    return activations
