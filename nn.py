import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score 
from torch.utils.data import Subset, DataLoader
from Dataset import get_dataset
from util import get_features_labels, plot_confusion_matrix
from feature_extractor import MelSpectrogramPCA, mfcc_transform

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

def evaluate_model(model, test_loader, device, plot=False):
    """
    Evaluates the model on a test set and prints classification metrics.
    
    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test set.
        device: Torch device (CPU or GPU).
        plot (bool): If True, plots the confusion matrix.
    """
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

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # Print classification report
    print("Final Test Performance:")
    print(classification_report(all_labels, all_preds))

    if plot: 
        plot_confusion_matrix(all_labels, all_preds, title="NN Confusion Matrix Mel Spectrogram")

    return accuracy, precision, recall, f1


def k_fold_cv(k, dataset, model_params, device, epochs=10):
    """
    Performs k-fold cross-validation for an audio classifier and prints average performance.

    Args:
        k (int): Number of folds.
        dataset: PyTorch Dataset containing audio data.
        model_params (dict): Parameters for initializing the model.
        device: Torch device (CPU or GPU).
        epochs (int): Number of training epochs.

    Returns:
        dict: Average accuracy, precision, recall, and F1-score across folds.
    """
    X, y = get_features_labels(dataset)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Metrics storage
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔹 Fold {fold + 1}/{k}")

        # Create train and validation subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=100, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=100, shuffle=False)

        # Initialize model, loss function, and optimizer
        model = AudioClassifier(**model_params).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train model
        fit(model, train_loader, criterion, optimizer, device, epochs, output=False)

        # Evaluate model on validation set
        acc, prec, rec, f1 = evaluate_model(model, val_loader, device)

        # Store metrics
        accuracy_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)

    # Compute average metrics
    avg_metrics = {
        "Accuracy": np.mean(accuracy_scores),
        "Precision": np.mean(precision_scores),
        "Recall": np.mean(recall_scores),
        "F1-score": np.mean(f1_scores)
    }

    # Print average results
    print("\nAverage Performance Across All Folds:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_directory = r"./Data/processed/train"
    test_directory = r"./Data/processed/test"
    sample_rate = 44100

    noise = False
    if input("Do you want to add noise? (y/n): ") == 'y':
        noise = True

    input_size=13*2
    transform = mfcc_transform
    
    if input("MFCC?: ") == 'n':
        pca_components=20
        transform = MelSpectrogramPCA(sample_rate=sample_rate, pca_components=pca_components)
        input_size=pca_components*2

    train_set = get_dataset(train_directory, sample_rate, transform=transform, noise_add=noise)
    test_set = get_dataset(test_directory, sample_rate, transform=transform, noise_add=noise)

    train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

    model_params={"input_size": input_size, "num_classes": 3, "hidden_layers": [128, 64]}
    model = AudioClassifier(**model_params).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    k_fold_cv(5, train_set, model_params=model_params, device=device, epochs=30)

    fit(model, train_loader, criterion, optimizer, device, epochs=30)
    evaluate_model(model, test_loader, device, plot=True)
