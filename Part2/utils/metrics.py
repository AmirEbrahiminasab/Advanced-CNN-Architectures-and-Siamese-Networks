import torch


def loss(out1, out2, y, m=1.0):
    """Function to calculate the loss"""
    dist = torch.norm(out1 - out2, p=2, dim=1)
    loss1 = y * (dist ** 2)
    loss2 = (1 - y) * torch.clamp(m ** 2 - dist ** 2, min=0)

    return torch.mean(loss1 + loss2)

