import numpy as np
import torch
from tqdm import tqdm


def embedded_dist(model, val_loader, device) -> tuple:
    """Function to get the embedding distance of similar/diff images"""
    model = model.model.to(device)
    model.eval()
    same_distances = []
    diff_distances = []
    count = 0

    with torch.no_grad():
        while True:
            for img1, img2, label in tqdm(val_loader):
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                feat1 = model(img1)
                feat2 = model(img2)
                distance = torch.norm(feat1 - feat2, p=2, dim=1).cpu().numpy()

                for d, l in zip(distance, label.cpu().numpy()):
                    if l == 1:
                        same_distances.append(d)
                    else:
                        diff_distances.append(d)
                    count += 1

                    if count >= 1000:
                        break

                del img1, img2, label
                if count >= 1000:
                    break
            if count >= 1000:
                break

    return same_distances, diff_distances


def get_neighbors(model, test_data, device) -> list:
    """Function to get the nearest neighbors"""
    test_set, test_path = test_data[0], np.array(test_data[1])
    model = model.model.to(device)
    model.eval()
    test_features = []
    with torch.no_grad():
        for img in tqdm(test_set):
            img = img.unsqueeze(0)
            img = img.to(device)
            feat = model(img).cpu().numpy()
            test_features.append(feat)
            del img

    test_features = np.concatenate(test_features, axis=0)
    samples_indices = np.random.choice(len(test_set), 5, replace=False)

    neighbors = []
    for idx in samples_indices:
        sample = test_features[idx]
        sample = sample.reshape(1, -1)
        dist = np.sqrt(np.sum((test_features - sample) ** 2, axis=1))

        sorted_indices = np.argsort(dist)
        nearest_indices = sorted_indices[1:11]

        neighbors.append({
            "sample_path": test_path[idx],
            "neighbor_paths": test_path[nearest_indices],
            "distances": dist[nearest_indices]
        })

    return neighbors


def get_features(model, test_set, device):
    """Function to return the features extracted from the model"""
    features = []
    model = model.model.to(device)
    model.eval()

    with torch.no_grad():
        for images in tqdm(test_set):
            images = images.to(device)
            output = model(images)
            features.append(output.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features
