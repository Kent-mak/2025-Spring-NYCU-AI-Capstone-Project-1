import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from Dataset import get_dataset
from torchaudio.transforms import MFCC
from util import get_features_labels
from feature_extractor import MelSpectrogramPCA, mfcc_transform


def evaluate_clustering(y_true, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

if __name__ == "__main__":
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


    train_features, train_labels = get_features_labels(train_set)
    test_features, test_labels = get_features_labels(test_set)
    
    # Standardize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Number of clusters is set to the number of unique labels (assuming prior knowledge)
    num_clusters = len(np.unique(train_labels))
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    train_cluster_labels = kmeans.fit_predict(train_features)
    test_cluster_labels = kmeans.predict(test_features)
    
    print("Clustering Performance on Training Data:")
    evaluate_clustering(train_labels, train_cluster_labels)
    
    print("Clustering Performance on Test Data:")
    evaluate_clustering(test_labels, test_cluster_labels)
