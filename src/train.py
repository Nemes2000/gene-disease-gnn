from torch_geometric.transforms import NormalizeFeatures
import torch

from .dataset import GeneDataset
from .model import GCN

dataset = GeneDataset(
    root="./data", 
    filenames=["gtex_genes.csv", "gene_graph.csv"],
    test_size=0.2,
    val_size=0.0,
    transform=NormalizeFeatures())
model = GCN(dataset[0], hidden_channels=16)

learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=decay)

criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad() 
      # Use all data as input, because all nodes have node features
      out = model(dataset)  
      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])  
      loss.backward() 
      optimizer.step()
      return loss

def test():
      model.eval()
      out = model(dataset)
      # Use the class with highest probability.
      pred = out.argmax(dim=1)  
      # Check against ground-truth labels.
      test_correct = pred[dataset.test_mask] == dataset.y[dataset.test_mask]  
      # Derive ratio of correct predictions.
      test_acc = int(test_correct.sum()) / int(dataset.test_mask.sum())  
      return test_acc

losses = []
for epoch in range(0, 1001):
    loss = train()
    losses.append(loss)
    if epoch % 100 == 0:
      print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')