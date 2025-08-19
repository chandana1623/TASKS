import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
# Example: Mall Customers dataset (if you have mall_customers.csv)
# df = pd.read_csv("Mall_Customers.csv")

# For demonstration, let’s simulate with sklearn’s make_blobs
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=200, n_features=2, centers=4, random_state=42)
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])

print(df.head())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
inertia = []  # sum of squared distances

K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertia, "bo-")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method for Optimal K")
plt.show()
# Suppose elbow suggests k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["Feature1"], y=df["Feature2"], hue=df["Cluster"], palette="Set2", s=60)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c="red", marker="X", label="Centroids")
plt.title("K-Means Clustering (K=4)")
plt.legend()
plt.show()
score = silhouette_score(X_scaled, df["Cluster"])
print("Silhouette Score:", score)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:,0]
df["PCA2"] = X_pca[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(x=df["PCA1"], y=df["PCA2"], hue=df["Cluster"], palette="Set1", s=60)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:,0], pca.transform(kmeans.cluster_centers_)[:,1],
            s=200, c="red", marker="X", label="Centroids")
plt.title("K-Means Clusters (PCA-reduced)")
plt.legend()
plt.show()
