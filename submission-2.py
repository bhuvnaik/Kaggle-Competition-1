

# XGBoost Ensemble with PCA & Node2Vec-style Relational Embedding
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from skopt import gp_minimize
from skopt.space import Real, Integer

BASE_PATH = "/kaggle/input/iisc-umc-301-kaggle-competition-1"
train = pd.read_csv(f"{BASE_PATH}/train.csv")
test = pd.read_csv(f"{BASE_PATH}/test.csv")

TARGET = "song_popularity"
X = train.drop(columns=["id", TARGET])
y = train[TARGET]
X_test = test.drop(columns=["id"])


def feature_engineering_base(df):
    df = df.copy()
    
    df = df.fillna(df.median(numeric_only=True))
    
    poly_cols = ["danceability","energy","audio_valence","tempo","loudness",
                 "acousticness","speechiness","instrumentalness"]
    existing_poly_cols = [c for c in poly_cols if c in df.columns]
    if existing_poly_cols:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_feats = poly.fit_transform(df[existing_poly_cols])
        poly_names = [f"poly_{c}" for c in poly.get_feature_names_out(existing_poly_cols)]
        poly_df = pd.DataFrame(poly_feats, columns=poly_names, index=df.index)
        df = pd.concat([df, poly_df], axis=1)
    
    pca_cols = ["energy","danceability","audio_valence","tempo","loudness",
                "acousticness","speechiness","instrumentalness"]
    existing_pca_cols = [c for c in pca_cols if c in df.columns]
    if existing_pca_cols:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[existing_pca_cols])
        pca = PCA(n_components=min(10, len(existing_pca_cols)), random_state=42)
        pca_feats = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(pca_feats, columns=[f"pca_emb_{i}" for i in range(pca_feats.shape[1])],
                              index=df.index)
        df = pd.concat([df, pca_df], axis=1)
    
    cluster_cols = ["energy","danceability","audio_valence","tempo","loudness"]
    existing_cluster_cols = [c for c in cluster_cols if c in df.columns]
    if existing_cluster_cols:
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df["song_cluster"] = kmeans.fit_predict(df[existing_cluster_cols])
    
    return df

def node2vec_fast(df, n_neighbors=6, embed_dim=8):
    df_feat = df.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_feat)
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    
    n_samples = X_scaled.shape[0]
    adjacency = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        adjacency[i, indices[i]] = 1
    adjacency = (adjacency + adjacency.T) / 2  # symmetric
    
    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adjacency_norm = adjacency / row_sums
    
    # SVD for embeddings
    u, s, vh = np.linalg.svd(adjacency_norm, full_matrices=False)
    node_embeddings = u[:, :embed_dim] * s[:embed_dim]
    
    embed_cols = [f"node2vec_emb_{i}" for i in range(embed_dim)]
    df_embed = pd.DataFrame(node_embeddings, columns=embed_cols, index=df_feat.index)
    df_feat = pd.concat([df_feat, df_embed], axis=1)
    
    return df_feat

X_full = pd.concat([X, X_test], axis=0, ignore_index=True)
X_full = feature_engineering_base(X_full)
X_full = node2vec_fast(X_full, n_neighbors=6, embed_dim=8)
X = X_full.iloc[:len(X)]
X_test = X_full.iloc[len(X):]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=27
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

# 17-XGBoost Ensemble with Bayesian Hyperparameter Tuning

val_preds = np.zeros(len(y_val))
test_preds = np.zeros(len(X_test))
model_weights = []

for model_idx in range(17):
    print(f"Training model {model_idx+1}/17...")
    
    def xgb_eval(params):
        param_dict = {
            'learning_rate': params[0],
            'max_depth': params[1],
            'subsample': params[2],
            'colsample_bytree': params[3],
            'gamma': params[4],
            'reg_alpha': params[5],
            'reg_lambda': params[6],
            'min_child_weight': params[7],
            'objective':'binary:logistic',
            'eval_metric':'auc'
        }
        model = xgb.train(
            param_dict,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain,"train"),(dval,"valid")],
            early_stopping_rounds=100,
            verbose_eval=False
        )
        val_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
        return -roc_auc_score(y_val, val_pred)
    
    from skopt import gp_minimize
    space = [
        Real(0.005,0.03,'log-uniform'),
        Integer(3,12),
        Real(0.6,0.95),
        Real(0.6,0.95),
        Real(0.0,0.05),
        Real(0.0,0.1),
        Real(0.8,1.5),
        Integer(1,10)
    ]
    
    res = gp_minimize(xgb_eval, space, n_calls=10, random_state=model_idx*11)
    
    best_params = {
        'learning_rate': res.x[0],
        'max_depth': res.x[1],
        'subsample': res.x[2],
        'colsample_bytree': res.x[3],
        'gamma': res.x[4],
        'reg_alpha': res.x[5],
        'reg_lambda': res.x[6],
        'min_child_weight': res.x[7],
        'objective':'binary:logistic',
        'eval_metric':'auc'
    }
    
    model = xgb.train(best_params, dtrain, num_boost_round=2000,
                      evals=[(dtrain,"train"),(dval,"valid")],
                      early_stopping_rounds=100,
                      verbose_eval=False)
    
    val_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
    test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration))
    
    auc = roc_auc_score(y_val, val_pred)
    model_weights.append(auc)
    val_preds += val_pred * auc
    test_preds += test_pred * auc

# Weighted ensemble
val_preds /= np.sum(model_weights)
test_preds /= np.sum(model_weights)

print("Validation AUC:", roc_auc_score(y_val, val_preds))
print("Validation Accuracy:", accuracy_score(y_val, (val_preds>0.5).astype(int)))

submission = pd.DataFrame({
    "id": test["id"],
    "song_popularity": test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")


