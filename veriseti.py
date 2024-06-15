import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def Veriseti():
    data = pd.read_csv('PeerIndex.csv')

    X = data.drop('Choice', axis=1)
    y = data['Choice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n")
    print("Logistic Regression")
    print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))
    print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n")
    print("RFC")
    print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))
    print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

