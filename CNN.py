import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Creating a CNN Model
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MushroomCNN(nn.Module):
    def __init__(self):
        super(MushroomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 10)  # Assuming 10 mushroom species

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
    

# Define Data Transforms
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# load existing modek
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////






# Option are either Train a model from scratch or load a pre existing model and further train it. both options aren not great and have benefit and flaws. In the short term lets use a pre existing model and further train it with some of the dtat




# Specify the path to your dataset on your computer
dataset_root = '/home/liam/git/Ai_Fungi_Finder/archive'

# Load datasets
train_dataset = datasets.ImageFolder(root=dataset_root, transform=data_transforms)
val_dataset = datasets.ImageFolder(root=dataset_root, transform=data_transforms)
test_dataset = datasets.ImageFolder(root=dataset_root, transform=data_transforms)

# Create Data Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)



# Instantiate Model, Loss Function, and Optimizer
model = MushroomCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        print("training")
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / total_samples
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy}')

# # Evaluation
# model.eval()
# with torch.no_grad():
#     total_correct = 0
#     total_samples = 0
#     for images, labels in test_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total_samples += labels.size(0)
#         total_correct += (predicted == labels).sum().item()
#     test_accuracy = total_correct / total_samples
#     print(f'Test Accuracy: {test_accuracy}')

# # Save Model (Optional)
# torch.save(model.state_dict(), 'mushroom_cnn.pth')
