import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class ArchiMixTrainer:
    def __init__(self):
        pass

    def __call__(self):
        pass

model = torch.load('saved_models/resnet101.pt')
print(model)