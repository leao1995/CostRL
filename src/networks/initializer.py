import torch
import torch.nn as nn
import logging

def initialize_weights(init_method, model):
    
    logging.info(f"\ninitialize network using {init_method}\n")
    
    if init_method == 'normal': # gaussian initialization
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)
    elif init_method == 'orthogonal': # orthogonal initialization
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
    elif init_method == 'zero':
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)