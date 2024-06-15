import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def Kumeleme():
    data = pd.read_csv('PeerIndex.csv')

    X = data.drop('Choice', axis=1)
    y = data['Choice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]}
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    X_cluster = X[['A_follower_count', 'A_following_count', 'A_listed_count', 'A_mentions_received', 'A_retweets_received', 'A_mentions_sent', 'A_retweets_sent', 'A_posts', 'A_network_feature_1', 'A_network_feature_2', 'A_network_feature_3']]

    kmeans = KMeans(n_clusters=2, random_state=42)
    X['cluster'] = kmeans.fit_predict(X_cluster)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='A_network_feature_1', y='A_network_feature_2', hue='cluster', data=X, palette='Set1', marker='X', s=150, edgecolor='black', linewidth=0.8)
    sns.scatterplot(x='A_network_feature_1', y='A_network_feature_2', hue=y_train, data=X_train, palette='viridis', alpha=0.6, s=80, edgecolor='w', linewidth=0.5)
    plt.title('Kümeleme Sonuçları ve Sınıflar')
    plt.show()
