import sys
import os
import numpy as np
import pickle
import yaml

import torch

np.random.seed(38)
import train
import evaluate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import model
from data import data_loader
from utils import visualization

imgs = data_loader.get_example()

visualization.visualize_image(imgs)

with open('../config/config.yaml', "r") as f:
    config = yaml.safe_load(f)


print(f"Training is on {'cuda' if torch.cuda.is_available() else 'cpu'}")

lowest_loss = float('inf')
best_model = None
best_hyperparameters = {'lr': None, 'gamma': None, 'step_size': None, 'batch_size': None, 'out_features': None}
best_state_dict = None
num_trials = 1
for _ in range(num_trials):
    lr = 10 ** np.random.uniform(-5, -1)
    gamma = np.random.uniform(0.01, 0.9)
    step_size = np.random.randint(2, 10)
    out_features = int(np.random.choice([256, 512]))
    batch_size = int(np.random.choice([8]))
    print(lr, gamma, step_size, batch_size, out_features)

    train_loader, val_loader, test_loader = data_loader.load_data(batch_size=batch_size)
    eff_model, train_loss, val_loss = train.train(
        out_features=out_features,
        train_loader=train_loader,
        val_loader=val_loader,
        step_size=step_size,
        gamma=gamma,
        learning_rate=lr,
        epochs=1
    )
    val_loss = val_loss[-1]

    if val_loss < lowest_loss:
        lowest_loss = val_loss
        best_model = eff_model
        best_hyperparameters['lr'] = lr
        best_hyperparameters['gamma'] = gamma
        best_hyperparameters['step_size'] = step_size
        best_hyperparameters['batch_size'] = batch_size
        best_hyperparameters['out_features'] = out_features
        best_state_dict = eff_model.state_dict()

print(f"Best loss on validation set: {lowest_loss:.3f}.")
print(f"Saving the best model with lr: {best_hyperparameters['lr']}, gamma: {best_hyperparameters['gamma']}, step_size: {best_hyperparameters['step_size']}, batch_size: {best_hyperparameters['batch_size']}, out_features: {best_hyperparameters['out_features']}")
# saving model
torch.save(best_state_dict, '../models/saved_models/best_model.pth')
# saving config
config['learning_rate'] = best_hyperparameters['lr']
config['step_size'] = best_hyperparameters['step_size']
config['out_features'] = best_hyperparameters['out_features']
config['gamma'] = best_hyperparameters['gamma']

with open('../config/config.yaml', "w") as f:
    yaml.dump(config, f)

train_loader, val_loader, test_loader = data_loader.load_data(batch_size=config['batch_size'])

best_model, train_loss, val_loss = train.train(
    out_features=config['out_features'],
    train_loader=train_loader,
    val_loader=val_loader,
    step_size=config['step_size'],
    gamma=config['gamma'],
    learning_rate=config['learning_rate'],
    epochs=1
)

# saving model
torch.save(best_model.state_dict(), '../models/saved_models/best_model.pth')
history = {"Train Loss": train_loss, "Val Loss": val_loss}
visualization.visualize_loss(history)
print("Saved the best model!")

# load model
best_model = model.Model(out_features=config['out_features'])
best_model.model.load_state_dict(torch.load('../models/saved_models/best_model.pth'))
print("Loaded the best model Successfully!")

# eval
same_distances, diff_distances = evaluate.embedded_dist(
                    model=best_model,
                    val_loader=val_loader,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

visualization.visualize_boxplot(same_distances, diff_distances)
print("Visualized BoxPlot Successfully!")

# neighbors
test_set = data_loader.get_test()
neighbors = evaluate.get_neighbors(best_model, test_set, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

visualization.visualize_neighbors(neighbors)
print("Visualized Neighbors Successfully!")

# visualize t-SNE
test_set, test_imgs = data_loader.get_sample()
features = evaluate.get_features(best_model, test_set, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
visualization.visualize_tsne(features, m)

# Bonus
best_model, train_loss, val_loss = train.train(
    out_features=config['out_features'],
    train_loader=train_loader,
    val_loader=val_loader,
    step_size=config['step_size'],
    gamma=config['gamma'],
    learning_rate=config['learning_rate'],
    epochs=1,
    m=2.0
)

# saving model
torch.save(best_model.state_dict(), '../models/saved_models/best_model1.pth')
history = {"Train Loss": train_loss, "Val Loss": val_loss}
visualization.visualize_loss(history, m='_with_m=2')
print("Saved the best model!")

# load model
best_model = model.Model(out_features=config['out_features'])
best_model.model.load_state_dict(torch.load('../models/saved_models/best_model1.pth'))
print("Loaded the best model Successfully!")

# eval
same_distances, diff_distances = evaluate.embedded_dist(
                    model=best_model,
                    val_loader=val_loader,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

visualization.visualize_boxplot(same_distances, diff_distances, m='_with_m=2')
print("Visualized BoxPlot Successfully!")

# neighbors
test_set = data_loader.get_test()
neighbors = evaluate.get_neighbors(best_model, test_set, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

visualization.visualize_neighbors(neighbors, m='_with_m=2')
print("Visualized Neighbors Successfully!")

# visualize t-SNE
test_set, test_imgs = data_loader.get_sample()
features = evaluate.get_features(best_model, test_set, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
visualization.visualize_tsne(features, m='_with_m=2')
