import torch
import torch.nn as nn

class FeedForwardNN (nn.Module):
    def __init__(self, hidden_layer_dimension, in_dim, drop_out_rate=0.3, activation_func=nn.ReLU()):
        super().__init__()
        layers = []
        current_in_dim = in_dim
        for current_out_dim in hidden_layer_dimension:
            layers.append(nn.Linear(current_in_dim,current_out_dim),activation_func, nn.Dropout(drop_out_rate)) #Adding all layers except last one
        layers.append(nn.Linear(current_out_dim,1))
        self.layer_container = nn.Sequential(*layers)
        
    def forward(self, X): 
        return self.layer_container(X).squeeze(-1)
        