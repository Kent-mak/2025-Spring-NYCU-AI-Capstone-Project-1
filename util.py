
from torchaudio.transforms import MFCC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def get_features_labels(dataset):

    features, labels = zip(*[dataset[i] for i in range(len(dataset))])
    features = np.stack(features)
    labels = np.array(labels)
    return features, labels

def plot_confusion_matrix(test_labels, predictions, title):
    # Compute confusion matrix
    conf_matrix = confusion_matrix(test_labels, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=set(test_labels), yticklabels=set(test_labels))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.show()
