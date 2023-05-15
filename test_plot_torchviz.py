## test _plot model
from torchviz import make_dot
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
a = torch.randn(3,5)
L = nn.Sequential(
    nn.Linear(5,5),
    nn.Tanh()
    )
dummy_out = L(a)

graph_filename = 'video_generator.pdf'
# model_dir = os.path.join('./datasets', 'train_models/ressim-256-cells/')
# model_dir = os.path.join('.\\datasets\\train_models\\ressim-256-cells\\')
fig_path = './'

graph_path = os.path.join(fig_path, graph_filename)
graph = make_dot(dummy_out, params=dict(L.named_parameters()))
# print(os.path.splitext(graph_path)[0])
# print(os.path.splitext(graph_path)[1][1:])
graph.render(os.path.splitext(graph_path)[0], format=os.path.splitext(graph_path)[1][1:])