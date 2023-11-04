import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Step 1: Generate the Spiral Dataset using NumPy
def generate_spiral_data(num_samples, num_classes):
    X = np.zeros((num_samples * num_classes, 2))
    y = np.zeros(num_samples * num_classes, dtype=int)
    for j in range(num_classes):
        ix = range(num_samples * j, num_samples * (j + 1))
        r = np.linspace(0.0, 1, num_samples)
        t = np.linspace(j * 4, (j + 1) * 4, num_samples) + np.random.randn(num_samples) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    return X, y

# Step 2: Define the Neural Network Model
class SpiralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SpiralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Step 3: Prepare the Data
num_samples = 300
num_classes = 3
X, y = generate_spiral_data(num_samples, num_classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class SpiralDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

train_dataset = SpiralDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Step 4: Define the Loss Function and Optimizer
input_size = 2
hidden_size = 100
num_epochs = 100
learning_rate = 0.01

model = SpiralNet(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Step 5: Train the Model
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        data, labels = batch['data'], batch['label']
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Step 6: Evaluate the Model
model.eval()
with torch.no_grad():
    X_test_tensor = torch.Tensor(X_test)
    predicted = model(X_test_tensor)
    _, predicted_labels = torch.max(predicted, 1)
    accuracy = (predicted_labels == torch.Tensor(y_test)).sum().item() / len(y_test) * 100
    print(f'Test Accuracy: {accuracy:.2f}%')

# Step 7: Plot the Decision Boundary
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
mesh_data = np.c_[xx.ravel(), yy.ravel()]
mesh_tensor = torch.Tensor(mesh_data)
with torch.no_grad():
    Z = model(mesh_tensor).detach().numpy()
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Spectral)
plt.show()
