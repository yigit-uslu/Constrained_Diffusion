import torch
import torch.nn as nn

class MLP(nn):
    def __init__(self, num_features_list):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(num_features_list)-1):
            layers.append(nn.Linear(num_features_list[i], num_features_list[i+1]))
            layers.append(nn.LeakyReLU(0.01))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)         
        return x