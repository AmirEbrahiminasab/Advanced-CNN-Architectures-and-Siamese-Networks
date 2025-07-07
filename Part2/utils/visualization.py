import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
matplotlib.use('Agg')


def visualize_image(imgs) -> None:
    """Function to visualize sample of faces"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        img = Image.open(imgs[i])
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('../utils/sample.png', dpi=350)


def visualize_loss(history, m='') -> None:
    """Function to visualize loss based on the dataset type"""
    plt.figure(figsize=(10, 6))

    plt.plot(history['Train Loss'], label='Train Loss', color='blue')
    plt.plot(history['Val Loss'], label='Val Loss', color='red')

    plt.title('Train & Val Loss over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(f'Loss', fontsize=14)
    plt.xticks(np.arange(1, 1+len(history['Val Loss']), step=2))
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(f'../utils/best_model_loss{m}.png', dpi=350)


def visualize_boxplot(same_distances, diff_distances, m='') -> None:
    """Function to visualize boxplot"""
    plt.figure(figsize=(10, 6))
    data = [same_distances, diff_distances]
    labels = ['Same Person (Label=1)', 'Different Persons (Label=0)']

    box = plt.boxplot(data, patch_artist=True, labels=labels)

    colors = ['#66b3ff', '#ff9999']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Feature Distance Distribution')
    plt.ylabel('Euclidean Distance')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(f'../utils/feature_boxplot{m}.png', dpi=350)


def visualize_neighbors(neighbors, m='') -> None:
    """Function to visualize nearest neighbors"""
    plt.figure(figsize=(30, 30))

    for i in range(5):
        sample = neighbors[i]['sample_path']
        plt.subplot(5, 11, i * 11 + 1)
        img = Image.open(sample)
        plt.imshow(img)
        plt.title("Chosen Image")
        plt.axis('off')

        for j in range(10):
            plt.subplot(5, 11, i * 11 + j + 2)
            img = Image.open(neighbors[i]['neighbor_paths'][j])
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Neighbor with dist: {neighbors[i]['distances'][j]:.2f}")

    plt.tight_layout()
    plt.savefig(f'../utils/nearest_neighbors{m}.png', dpi=350)


def visualize_tsne(features, m=''):
    """Function to visualize t-SNE"""
    tsne = TSNE(n_components=2, random_state=38, perplexity=5)
    features_tsne = tsne.fit_transform(features)

    plt.figure(figsize=(12, 8))
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    unique_labels = []
    handles = []

    for i in range(15):
        person = i // 3
        label = f'Person {person + 1}'

        if label not in unique_labels:
            unique_labels.append(label)
            handles.append(plt.scatter(features_tsne[i, 0],
                                       features_tsne[i, 1],
                                       c=colors[person],
                                       label=label,
                                       alpha=0.7,
                                       s=100))
        else:
            plt.scatter(features_tsne[i, 0],
                        features_tsne[i, 1],
                        c=colors[person],
                        alpha=0.7,
                        s=100)
    plt.legend(handles=handles, fontsize=12)

    plt.title('t-SNE Visualization of Feature Vectors', fontsize=14)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.savefig(f'../utils/tsne{m}.png', dpi=350)
