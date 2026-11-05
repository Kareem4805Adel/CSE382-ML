import optuna
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.preprocessing.preprocess import load_and_preprocess_clustering

X, df_raw = load_and_preprocess_clustering()


def objective(trial):
    k = trial.suggest_int("n_clusters", 2, 10)
    n_init = trial.suggest_categorical("n_init", [10, 20])
    model = KMeans(n_clusters=k, n_init=n_init, random_state=42)
    model.fit(X)
    return silhouette_score(X, model.labels_, sample_size=10000, random_state=42)


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20)

best_k = study.best_params["n_clusters"]
best_n_init = study.best_params["n_init"]

best_model = KMeans(n_clusters=best_k, n_init=best_n_init, random_state=42)
best_model.fit(X)
dbi = davies_bouldin_score(X, best_model.labels_)

print("Optimized KMeans (Optuna)")
print("best params =", study.best_params)
print("silhouette =", study.best_value)
print("davies-bouldin =", dbi)