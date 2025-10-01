from captum.attr import IntegratedGradients
import torch

from config import Config
from models.weight_model import Weight

model = Weight(4, Config.v_emb_dim, 1, Config.v_act_type)
model.load_state_dict(torch.load("1073_vnet_weights.pth"))
model.eval()

# tételezzük fel, hogy van egy input: x (1, num_features)
x = torch.randn(1, 4, requires_grad=True)

# attribúció számítása
ig = IntegratedGradients(model)
attr, delta = ig.attribute(x, target=0, return_convergence_delta=True)

print("Feature importance scores:", attr.squeeze().detach().cpu().numpy())