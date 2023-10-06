import torch
import torch.nn as nn
from sklearn.preprocessing import Normalizer, StandardScaler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

# Predict day type based off of previous days high/low/close etc

# Tensorboard Writer object
writer = SummaryWriter("runs\iter1")

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
input_size = 8  # No. of input values
hidden_size = 10  # size of hidden layers
output_size = 8  # output layer size
num_epochs = 100  # training iterations
learning_rate = 0.01  # model learning rate
batch_size = 1  # training input/label pairs per batch
patience = 5  # tolerance for validation early stop
FILE = "trained_model.pth"  # path to save model state

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

# Data for tensorboard
examples = iter(test_loader)
examples_data, examples_labels = next(examples)
examples_data.to(device)


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

# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Tensorboard model graph
writer.add_graph(model.to(device), examples_data.to(device))

# Training Loop
total_steps = len(train_loader)

# For tensorboard
running_loss = 0.0

# For validation
patience_counter = 0
best_loss = 0.0
best_model = model.state_dict()


for epoch in range(num_epochs):

    # Training
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = torch.squeeze(labels, dim=1)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1}/{total_steps}, loss = {loss.item():.4f}')

            # Tensorboard running loss
            writer.add_scalar('training loss', running_loss / 100, epoch * total_steps + i)
            running_loss = 0.0

    # Validation and early stopping
    model.eval()
    temp_loss = 0.0
    for i, (inputs, labels) in enumerate(test_loader):
        # Calculate loss
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = torch.squeeze(labels, dim=1)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Sum loss
        temp_loss += loss

    # Assign initial loss
    if best_loss == 0.0:
        best_loss = temp_loss

    # Updated best loss if better
    if temp_loss < best_loss:
        best_loss = temp_loss
        best_model = model.state_dict()
        patience_counter = 0

    # Increase patience counter if worse
    if temp_loss > best_loss:
        patience_counter += 1

    # Save model and stop training if patience reached
    if patience_counter >= patience:
        torch.save(model.state_dict(), FILE)
        exit(1)








