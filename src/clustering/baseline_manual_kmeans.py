from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.preprocessing.preprocess import load_and_preprocess_clustering

X, df_raw = load_and_preprocess_clustering()

model = KMeans(n_clusters=4, n_init=10, random_state=42)
model.fit(X)

sil = silhouette_score(X, model.labels_, sample_size=10000, random_state=42)
dbi = davies_bouldin_score(X, model.labels_)

print("Manual KMeans")
print("k =", 4)
print("silhouette =", sil)
print("davies-bouldin =", dbi)