import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from Dataset import get_dataset
from torchaudio.transforms import MFCC
from util import get_features_labels, plot_confusion_matrix
from feature_extractor import MelSpectrogramPCA, mfcc_transform



def k_fold_cv(k, X, y, model_params):
    if model_params is None:
        model_params = {"kernel": "rbf", "C": 1.0, "gamma": "scale"}

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

     # Metrics storage
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{k}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train SVM
        svm_model = SVC(**model_params)
        svm_model.fit(X_train, y_train)

        # Predictions
        val_predictions = svm_model.predict(X_val)

        # Compute metrics
        acc = accuracy_score(y_val, val_predictions)
        prec = precision_score(y_val, val_predictions, average="weighted", zero_division=0)
        rec = recall_score(y_val, val_predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_val, val_predictions, average="weighted", zero_division=0)

        # Store metrics
        accuracy_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)

        # Print fold-wise classification report
        print(classification_report(y_val, val_predictions))

    # Compute average performance
    avg_metrics = {
        "Accuracy": np.mean(accuracy_scores),
        "Precision": np.mean(precision_scores),
        "Recall": np.mean(recall_scores),
        "F1-score": np.mean(f1_scores)
    }

    # Print summary
    print("\nAverage Performance Across All Folds:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")



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
    
    print(train_features.shape)
    # Standardize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    model_params={"kernel": "rbf", "C": 1.0, "gamma": "scale"}

    k_fold_cv(5, X=train_features, y=train_labels, model_params=model_params)

    svm_model = SVC(**model_params)

    # Final training on full dataset
    svm_model.fit(train_features, train_labels)
    
    # Make predictions on the test set
    predictions = svm_model.predict(test_features)

    # Evaluate accuracy, recall, precision, and F1-score
    print("Final Test Performance:")
    print(classification_report(test_labels, predictions))

    plot_confusion_matrix(test_labels, predictions, title="SVM Confusion Matrix Mel Spectrogram")
