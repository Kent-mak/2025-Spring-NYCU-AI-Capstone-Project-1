import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from torch.utils.data import Subset, DataLoader
from Dataset import get_dataset
from util import get_features_labels
from torchaudio.transforms import MFCC

# Define a simple neural network classifier
class AudioClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers=[128, 64]):
        """
        Args:
            input_size (int): Number of input features.
            num_classes (int): Number of output classes.
            hidden_layers (list): List of hidden layer sizes.
        """
        super(AudioClassifier, self).__init__()
        
        # Define the layers dynamically
        layers = []
        prev_size = input_size  # Initial input size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())  # Activation
            prev_size = hidden_size  # Update size for next layer
        
        layers.append(nn.Linear(prev_size, num_classes))  # Final layer
        layers.append(nn.LogSoftmax(dim=1))  # Output activation
        
        self.model = nn.Sequential(*layers)  # Store as a single Sequential module

    def forward(self, x):
        return self.model(x)
    

def standardize(x):
    mean = x.mean(dim=0, keepdim=True)  # Mean across batch dimension
    std = x.std(dim=0, keepdim=True)
    std[std == 0] = 1e-8  # Avoid division by zero
    return (x - mean) / std

def fit(model, train_loader, criterion, optimizer, device, epochs=10, output=True): 
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # print(inputs.shape)
            inputs = standardize(inputs)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if output:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = standardize(inputs)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("Final Test Performance:")
    print(classification_report(all_labels, all_preds))

def k_fold_cv(k, dataset, model_params, device, epochs=10):

    X, y = get_features_labels(dataset)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{k}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=100, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=100, shuffle=False)
        # print(y_train.shape)

        model = AudioClassifier(**model_params).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        fit(model, train_loader, criterion, optimizer, device, epochs, output=False)

        evaluate_model(model, val_loader, device)
        


mfcc_transform = MFCC(
    sample_rate=44100,  # Match dataset sample rate
    n_mfcc=13,  # Number of MFCC coefficients
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_directory = r"./Data/processed/train"
    test_directory = r"./Data/processed/test"
    sample_rate = 44100

    noise = False
    if input("Do you want to add noise? (y/n): ") == 'y':
        noise = True

    train_set = get_dataset(train_directory, sample_rate, transform=mfcc_transform, noise=noise)
    test_set = get_dataset(test_directory, sample_rate, transform=mfcc_transform, noise=noise)

    train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

    model_params={"input_size": 13, "num_classes": 3, "hidden_layers": [128, 64]}
    model = AudioClassifier(**model_params).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    k_fold_cv(5, train_set, model_params=model_params, device=device, epochs=20)

    fit(model, train_loader, criterion, optimizer, device, epochs=20)
    evaluate_model(model, test_loader, device)
