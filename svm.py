import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from Dataset import get_dataset
from torchaudio.transforms import MFCC
from util import get_features_labels


mfcc_transform = MFCC(
    sample_rate=44100,  # Match dataset sample rate
    n_mfcc=13,  # Number of MFCC coefficients
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
)

def k_fold_cv(k, X, y, model_params):
    if model_params is None:
        model_params = {"kernel": "rbf", "C": 1.0, "gamma": "scale"}

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{k}")
        X_train,  y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # print(y_train.shape)

        svm_model = SVC(**model_params)
        svm_model.fit(X_train, y_train)

        val_predictions = svm_model.predict(X_val)
        print(classification_report(y_val, val_predictions))



if __name__ == "__main__":
    train_directory = r"./Data/processed/train"
    test_directory = r"./Data/processed/test"
    sample_rate = 44100

    noise = False
    if input("Do you want to add noise? (y/n): ") == 'y':
        noise = True

    train_set = get_dataset(train_directory, sample_rate, transform=mfcc_transform, noise=noise)
    test_set = get_dataset(test_directory, sample_rate, transform=mfcc_transform, noise=noise)


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
