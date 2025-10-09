import torch
from captum.attr import IntegratedGradients
from config import Config
from models.weight_model import Weight
import numpy as np

device = torch.device('cpu')

model = Weight(4, Config.v_emb_dim, 1, Config.v_act_type)
state_dict = torch.load("results/multitask/807/807_vnet_weights.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
ig = IntegratedGradients(model)
n_runs = 100
attr_list = []

for i in range(n_runs):
    x = torch.randn(1, 4, requires_grad=True, device=device)
    attr, delta = ig.attribute(x, target=0, return_convergence_delta=True)
    attr_np = attr.squeeze().detach().cpu().numpy()
    attr_list.append(attr_np)
    #print(f"Run {i+1} attribution:", attr_np)

# Convert to numpy array for stats
attr_array = np.stack(attr_list)

# Compute mean and std
attr_mean = attr_array.mean(axis=0)
attr_std = attr_array.std(axis=0)

print("\n=== Summary across runs ===")
print("Mean feature importance:", attr_mean)
print("Std feature importance:", attr_std)
