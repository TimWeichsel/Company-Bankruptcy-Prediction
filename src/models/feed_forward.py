import torch
import torch.nn as nn

class FeedForwardNN (nn.Module):
    def __init__(self, in_dim, hidden_layer_dimension = (64, 128), dropout_rate=0.3, activation_func_class=nn.ReLU):
        super().__init__()
        layers = []
        current_in_dim = in_dim
        for current_out_dim in hidden_layer_dimension:
            layers += [nn.Linear(current_in_dim,current_out_dim),activation_func_class(), nn.Dropout(dropout_rate)] #Adding all layers except last one
            current_in_dim = current_out_dim
        layers += [nn.Linear(current_in_dim,1)] #Adding Last layer
        self.layer_container = nn.Sequential(*layers)
        
    def forward(self, X): 
        return self.layer_container(X).squeeze(-1)
        