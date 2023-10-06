import torch
import torch.nn as nn
from sklearn.preprocessing import Normalizer, StandardScaler
from torch.utils.data import Dataset
import pandas as pd

# Predict day type based off of previous days high/low/close etc

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
input_size = 8  # No. of input values
hidden_size = 10  # size of hidden layers
output_size = 8  # output layer size
batch_size = 1  # training input/label pairs per batch


# Load Data
class CSVDataset(Dataset):
    def __init__(self, file_name, data_index, label_index):
        # Load X/y Data from .csv file
        file_out = pd.read_csv(file_name)
        y = file_out[label_index]
        x = file_out[data_index]

        # Scale features and assign to attributes
        sc = StandardScaler()
        self.X = sc.fit_transform(x)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x_train = torch.tensor(self.X[item], dtype=torch.float32)
        y_train = torch.tensor(self.y.iloc[item])
        return x_train, y_train


dataset = CSVDataset("profile_data.csv", ["FH_High", "FH_Low", "FH_BidVol", "FH_AskVol", "D_Low", "D_Close",
                                          "D_BidVol", "D_AskVol"], ["labels"])

# Splitting into train/test and creating iterable object
torch.manual_seed(0)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# Data loader iterable creation
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)


# Model
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


# Model Implementation
model = NeuralNet(input_size, hidden_size, output_size)
model = model.to(device)
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

correct = 0
total = 0
for i, (inputs, labels) in enumerate(loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    outputs = torch.exp(outputs)
    _, predicted = torch.max(outputs, 1)
    if predicted == 2:
        print(outputs, predicted.item(), labels.item())
        if predicted.item() == labels.item():
            correct += 1
        total += 1
print(f'Total Accuracy: {(correct / total * 100):.2f}%')

