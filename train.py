from torch.utils.data import DataLoader
import numpy as np
import os
from Dataset import get_dataloaders




def main():
    train_directory = r".\Data\processed\train"
    test_directory = r".\Data\processed\test"
    sample_rate = 44100

    train_loader, val_loader = get_dataloaders(train_directory, sample_rate, train=True)
    test_loader = get_dataloaders(test_directory, sample_rate, train=False)

if __name__ == "__main__":
    main()