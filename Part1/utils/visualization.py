import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import numpy as np
from PIL import Image
matplotlib.use('Agg')


def visualize_augmentation(original_img, augmented_color, augmented_affine) -> None:
    """Function to visualize augmentations"""
    images = [original_img] + augmented_color + augmented_affine
    titles = ["Original"] + [f"Color Jitter {i + 1}" for i in range(10)] + [f"Random Affine {i + 1}" for i in range(10)]

    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(4, 6, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig('../utils/augmentation.png', dpi=350)


def visualize_loss(history, model_type) -> None:
    """Function to visualize loss during training"""
    plt.figure(figsize=(10, 6))

    plt.plot(history['Train Loss'], label='Train Loss', color='blue')
    plt.plot(history['Val Loss'], label='Val Loss', color='red')

    plt.title(f'Train & Val Loss over Epochs Model with Block {model_type}', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(f'Loss', fontsize=14)
    plt.xticks(np.arange(1, 1+len(history['Val Loss']), step=2))
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(f'../utils/model_{model_type}_loss.png', dpi=350)


def visualize_acc(history, model_type) -> None:
    """Function to visualize accuracy during training"""
    plt.figure(figsize=(10, 6))

    plt.plot(history['Train Accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history['Val Accuracy'], label='Val Accuracy', color='red')

    plt.title(f'Train & Val Accuracy over Epochs Model with Block {model_type}', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(f'Accuracy', fontsize=14)
    plt.xticks(np.arange(1, 1+len(history['Val Accuracy']), step=2))
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(f'../utils/model_{model_type}_acc.png', dpi=350)


def visualize_loss_acc(history, model_type) -> None:
    """General Function to visualize Loss and Accuracy"""
    visualize_loss(history, model_type)
    visualize_acc(history, model_type)


def plot_confusion_matrix(y_true, y_pred, model_type, labels_map):
    """Function for plotting the confusion matrix based on the model type"""
    cm = confusion_matrix(y_true, y_pred, labels=list(labels_map.keys()))
    labels = [labels_map[label] for label in sorted(labels_map.keys())]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'../utils/Confusion_Matrix_{model_type}.png', dpi=350)


def visualize_tsne(embeddings, labels, model_type, label_map):
    """Function to visualize the t-SNE of model embedding outputs"""
    tsne = TSNE(n_components=2, random_state=38)
    embeddings_2d = tsne.fit_transform(embeddings)
    labels = np.array([label_map[label] for label in labels])

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=labels,
        palette=sns.color_palette("hsv", n_colors=len(np.unique(labels))),
        legend="full",
        alpha=0.6
    )
    plt.title(f't-SNE Visualization of Embeddings of Model {model_type}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.savefig(f'../utils/TSNE_{model_type}.png', dpi=350)


def visualization_layer(activations):
    """Function to visualize layers output"""
    plt.figure(figsize=(15, 10))
    for idx, act in enumerate(activations):
        # for skipping linear layers
        if act.dim() != 4:
            continue

        max_act = act.max(dim=1).values.squeeze(0).cpu().numpy()
        max_act = (max_act - max_act.min()) / (max_act.max() - max_act.min() + 1e-8)

        plt.subplot(4, 5, idx + 1)
        plt.imshow(max_act, cmap='viridis')
        plt.title(f'Layer {idx + 1}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'../utils/layer_by_layer.png', dpi=350)


def visualize_attention(model, test_dataset):
    """Function to visualize the attention of embedded outputs for Block B"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modell = model.to(device)
    modell.eval()
    feature_extractor = nn.Sequential(*list(modell.children())[:12])

    activation = {}

    def hook_fn(module, input, output):
        activation['pre_sigmoid'] = output

    hook = modell.block10.conv1.register_forward_hook(hook_fn)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    images, labels = [], []
    for i, (img, lbl) in enumerate(test_loader):
        if i >= 10:
            break
        images.append(img)
        labels.append(lbl.item())

    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]
    denorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    plt.figure(figsize=(20, 15))
    for i in range(10):
        img_tensor = images[i].to(device)
        original_img = denorm(img_tensor.squeeze()).permute(1, 2, 0).cpu().numpy()
        original_img = np.clip(original_img, 0, 1)

        with torch.no_grad():
            _ = feature_extractor(img_tensor)

        pre_sigmoid = activation['pre_sigmoid']
        attn_map = torch.sigmoid(pre_sigmoid).squeeze().cpu().numpy()

        original_size = img_tensor.shape[-2:]
        attn_map_upsampled = F.interpolate(
            torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0),
            size=original_size,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        attn_map_norm = (attn_map_upsampled - attn_map_upsampled.min()) / (
                    attn_map_upsampled.max() - attn_map_upsampled.min())
        heatmap = plt.get_cmap('jet')(attn_map_norm)
        heatmap = np.uint8(heatmap * 255)
        heatmap = heatmap[:, :, :3]

        original_img_uint8 = np.uint8(original_img * 255)
        overlay = cv2.addWeighted(original_img_uint8, 0.6, heatmap, 0.4, 0)

        plt.subplot(4, 5, 2 * i + 1)
        plt.imshow(original_img)
        plt.title(f'Original\nLabel: {labels[i]}')
        plt.axis('off')

        plt.subplot(4, 5, 2 * i + 2)
        plt.imshow(overlay)
        plt.title('Attention Overlay')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'../utils/Sigmoid_BlockB.png', dpi=350)
    hook.remove()
