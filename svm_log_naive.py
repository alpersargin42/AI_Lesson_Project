import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pandas as pd


def Svm_log_naive():
    data = pd.read_csv('PeerIndex.csv')

    X = data.drop('Choice', axis=1)
    y = data['Choice']

    kfold_values = [3, 5, 7, 10]

    cv_scores_lr = []
    for kfold in kfold_values:
        lr_model = LogisticRegression()
        scores = cross_val_score(lr_model, X, y, cv=kfold)
        mean_score = np.mean(scores)
        cv_scores_lr.append(mean_score)
        print(f"Logistic Regression with k-fold={kfold} - Mean Accuracy: {mean_score:.4f}")

    cv_scores_svm = []
    for kfold in kfold_values:
        svm_model = SVC()
        scores = cross_val_score(svm_model, X, y, cv=kfold)
        mean_score = np.mean(scores)
        cv_scores_svm.append(mean_score)
        print(f"SVM with k-fold={kfold} - Mean Accuracy: {mean_score:.4f}")

    cv_scores_nb = []
    for kfold in kfold_values:
        nb_model = GaussianNB()
        scores = cross_val_score(nb_model, X, y, cv=kfold)
        mean_score = np.mean(scores)
        cv_scores_nb.append(mean_score)
        print(f"Naive Bayes with k-fold={kfold} - Mean Accuracy: {mean_score:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(kfold_values, cv_scores_lr, marker='o', linestyle='-', label='Logistic Regression')
    plt.plot(kfold_values, cv_scores_svm, marker='o', linestyle='-', label='SVM')
    plt.plot(kfold_values, cv_scores_nb, marker='o', linestyle='-', label='Naive Bayes')
    plt.title('Cross-Validation Score for Different k-fold Values')
    plt.xlabel('k-fold Value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()