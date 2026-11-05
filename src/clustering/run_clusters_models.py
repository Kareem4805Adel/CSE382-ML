import pandas as pd
import baseline_manual_kmeans
import optimized_kmeans

table = pd.DataFrame({
    "model": ["manual", "optimized (Optuna)"],
    "k": [baseline_manual_kmeans.model.n_clusters, optimized_kmeans.best_k],

    "silhouette": [baseline_manual_kmeans.sil, optimized_kmeans.study.best_value],

    "davies_bouldin": [baseline_manual_kmeans.dbi, optimized_kmeans.dbi],
})

print(table)