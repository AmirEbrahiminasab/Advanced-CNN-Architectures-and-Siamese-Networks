import sys
import os
import numpy as np
from multiprocessing import freeze_support
import pickle
import yaml

import torch
import torch.nn as nn

np.random.seed(38)
import train
import evaluate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import model
from data import data_loader
from utils import visualization


with open('../config/config.yaml', "r") as f:
    config = yaml.safe_load(f)

if __name__ == '__main__':
    freeze_support()
    train_loader, val_loader, test_loader = data_loader.load_data(batch_size=64)
    print("Loaded DataLoader Successfully!")
    label_map = data_loader.get_label_map()
    # visualize augmentations
    original_img, augmented_color, augmented_affine = data_loader.sample()
    visualization.visualize_augmentation(original_img, augmented_color, augmented_affine)

    # model with BlockA
    modell, history = train.train(train_loader, val_loader, model.BlockA, epochs=1)
    torch.save(modell.state_dict(), "../models/saved_models/model_a.pth")
    visualization.visualize_loss_acc(history, 'A')
    modell = model.Model(model.BlockA)
    print(f"Model_A parameters: {model.get_parameters(modell)}")
    modell.load_state_dict(torch.load('../models/saved_models/model_a.pth'))
    y_pred, y_true = evaluate.pred(test_loader, modell)
    visualization.plot_confusion_matrix(y_true, y_pred, 'A', label_map)
    feature_extractor = nn.Sequential(*list(modell.children())[:15])
    embeddings, labels = evaluate.embedding(test_loader, feature_extractor)
    visualization.visualize_tsne(embeddings, labels, 'A', label_map)

    # model with BlockB
    modell, history = train.train(train_loader, val_loader, model.BlockB, epochs=1)
    torch.save(modell.state_dict(), "../models/saved_models/model_b.pth")
    visualization.visualize_loss_acc(history, 'B')
    modell = model.Model(model.BlockB)
    print(f"Model_B parameters: {model.get_parameters(modell)}")
    modell.load_state_dict(torch.load('../models/saved_models/model_b.pth'))
    y_pred, y_true = evaluate.pred(test_loader, modell)
    visualization.plot_confusion_matrix(y_true, y_pred, 'B', label_map)
    feature_extractor = nn.Sequential(*list(modell.children())[:15])
    embeddings, labels = evaluate.embedding(test_loader, feature_extractor)
    visualization.visualize_tsne(embeddings, labels, 'B', label_map)
    visualization.visualize_attention(modell, data_loader.get_test())

    # model with BlockC
    modell, history = train.train(train_loader, val_loader, model.BlockC, epochs=1)
    torch.save(modell.state_dict(), "../models/saved_models/model_c.pth")
    visualization.visualize_loss_acc(history, 'C')
    modell = model.Model(model.BlockC)
    print(f"Model_C parameters: {model.get_parameters(modell)}")
    modell.load_state_dict(torch.load('../models/saved_models/model_c.pth'))
    y_pred, y_true = evaluate.pred(test_loader, modell)
    visualization.plot_confusion_matrix(y_true, y_pred, 'C', label_map)
    feature_extractor = nn.Sequential(*list(modell.children())[:15])
    embeddings, labels = evaluate.embedding(test_loader, feature_extractor)
    visualization.visualize_tsne(embeddings, labels, 'C', label_map)

    # model with BlockD
    modell, history = train.train(train_loader, val_loader, model.BlockD, epochs=1)
    torch.save(modell.state_dict(), "../models/saved_models/model_d.pth")
    visualization.visualize_loss_acc(history, 'D')
    modell = model.Model(model.BlockD)
    print(f"Model_D parameters: {model.get_parameters(modell)}")
    modell.load_state_dict(torch.load('../models/saved_models/model_d.pth'))
    y_pred, y_true = evaluate.pred(test_loader, modell)
    visualization.plot_confusion_matrix(y_true, y_pred, 'D', label_map)
    feature_extractor = nn.Sequential(*list(modell.children())[:15])
    embeddings, labels = evaluate.embedding(test_loader, feature_extractor)
    visualization.visualize_tsne(embeddings, labels, 'D', label_map)

    # model with BlockDw
    modell, history = train.train(train_loader, val_loader, model.DepthwiseSepD, epochs=1)
    torch.save(modell.state_dict(), "../models/saved_models/model_dw.pth")
    visualization.visualize_loss_acc(history, 'D Depthwise Separable')
    modell = model.Model(model.DepthwiseSepD)
    print(f"Model_D depthwise parameters: {model.get_parameters(modell)}")
    modell.load_state_dict(torch.load('../models/saved_models/model_dw.pth'))
    y_pred, y_true = evaluate.pred(test_loader, modell)
    visualization.plot_confusion_matrix(y_true, y_pred, 'D Depthwise Separable', label_map)
    feature_extractor = nn.Sequential(*list(modell.children())[:15])
    embeddings, labels = evaluate.embedding(test_loader, feature_extractor)
    visualization.visualize_tsne(embeddings, labels, 'D Depthwise Separable', label_map)

    # model with BlockE
    modell, history = train.train(train_loader, val_loader, model.BlockE, epochs=1)
    torch.save(modell.state_dict(), "../models/saved_models/model_e.pth")
    visualization.visualize_loss_acc(history, 'E')
    modell = model.Model(model.BlockE)
    print(f"Model_E parameters: {model.get_parameters(modell)}")
    modell.load_state_dict(torch.load('../models/saved_models/model_e.pth'))
    y_pred, y_true = evaluate.pred(test_loader, modell)
    visualization.plot_confusion_matrix(y_true, y_pred, 'E', label_map)
    feature_extractor = nn.Sequential(*list(modell.children())[:15])
    embeddings, labels = evaluate.embedding(test_loader, feature_extractor)
    visualization.visualize_tsne(embeddings, labels, 'E', label_map)
    # visualize last model layer by layer
    activations = evaluate.layer_by_layer(data_loader.get_sample_test_img(), modell)
    visualization.visualization_layer(activations)

