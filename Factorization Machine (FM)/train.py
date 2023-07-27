import torch
import torch.optim as optim

from dataset import FMDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

dataset = FMDataset()
train_dataset, test_dataset = train_test_split(dataset, train_size=0.8, random_state=42, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = FactorizationMachine()
optimizer = optim.Adam()



