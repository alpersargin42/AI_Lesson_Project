import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd

def Roc():
    data = pd.read_csv('PeerIndex.csv')

    X = data.drop('Choice', axis=1)
    y = data['Choice']

    y_bin = label_binarize(y, classes=[0, 1, 2])

    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

    lr_model = OneVsRestClassifier(LogisticRegression())
    lr_model.fit(X_train, y_train)

    y_score = lr_model.decision_function(X_test)

    plt.figure(figsize=(10, 6))
    for i in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Logistic Regression')
    plt.legend()
    plt.grid(True)
    plt.show()
