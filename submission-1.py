

# Spectral Feature XGBoost Ensemble
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cluster import KMeans

RANDOM_STATE = 42
TARGET = "song_popularity"
TRAIN_PATH = "/kaggle/input/iisc-umc-301-kaggle-competition-1/train.csv"
TEST_PATH  = "/kaggle/input/iisc-umc-301-kaggle-competition-1/test.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

X = train.drop(columns=["id", TARGET])
y = train[TARGET].values
X_test = test.drop(columns=["id"])

def spectral_feature_engineering(df):
    df = df.copy()

    if "song_duration_ms" in df.columns:
        df["duration_min"] = df["song_duration_ms"] / 60000
        df["duration_log"] = np.log1p(df["song_duration_ms"].clip(lower=0))

    for col in ["tempo", "loudness"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(np.abs(df[col]))

    if {"speechiness", "instrumentalness"}.issubset(df.columns):
        df["speech_instr_ratio"] = df["speechiness"] / (df["instrumentalness"] + 1e-6)
    if {"energy", "danceability"}.issubset(df.columns):
        df["energy_dance_ratio"] = df["energy"] * df["danceability"]
    if {"energy", "loudness"}.issubset(df.columns):
        df["energy_loudness_ratio"] = df["energy"] / (np.abs(df["loudness"]) + 1e-6)
    if {"acousticness", "danceability"}.issubset(df.columns):
        df["acoustic_dance_ratio"] = df["acousticness"] / (df["danceability"] + 1e-6)
    if {"audio_valence", "energy"}.issubset(df.columns):
        df["valence_energy_ratio"] = df["audio_valence"] * df["energy"]

    rank_cols = ["tempo", "loudness", "energy", "danceability"]
    for col in rank_cols:
        if col in df.columns:
            df[f"{col}_rank"] = df[col].rank(pct=True)

    cluster_cols = ["danceability", "energy", "loudness", "tempo", "acousticness"]
    existing_cluster_cols = [c for c in cluster_cols if c in df.columns]
    if existing_cluster_cols:
        cluster_input = df[existing_cluster_cols].fillna(0.0)
        kmeans = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10)
        df["song_cluster"] = kmeans.fit_predict(cluster_input)

    poly_cols = ["danceability", "energy", "audio_valence", "tempo", "loudness"]
    existing_poly_cols = [c for c in poly_cols if c in df.columns]
    if existing_poly_cols:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_input = df[existing_poly_cols].fillna(0.0)
        poly_feats = poly.fit_transform(poly_input)
        poly_names = [f"poly_{c}" for c in poly.get_feature_names_out(existing_poly_cols)]
        poly_df = pd.DataFrame(poly_feats, columns=poly_names, index=df.index)
        df = pd.concat([df, poly_df], axis=1)

    return df

X = spectral_feature_engineering(X)
X_test = spectral_feature_engineering(X_test)

print("Final Train Shape:", X.shape)
print("Final Test Shape:", X_test.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val, label=y_val)
dtest  = xgb.DMatrix(X_test)

# 17 XGBOOST MODELS 
param_list = [
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.005, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 11},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.006, "max_depth": 8, "subsample": 0.85,"colsample_bytree": 0.75,"random_state": 12},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.003, "max_depth": 10,"subsample": 0.9,"colsample_bytree": 0.9,"random_state": 41},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.02, "max_depth": 3, "subsample": 0.7, "colsample_bytree": 0.7,"random_state": 14},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.015,"max_depth": 4, "subsample": 0.65,"colsample_bytree": 0.9,"random_state": 15},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.01, "max_depth": 6, "subsample": 0.95,"colsample_bytree": 0.65,"random_state": 16},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.015,"max_depth": 7, "subsample": 0.75,"colsample_bytree": 0.85,"random_state": 17},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.03, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8,"random_state": 38},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.025,"max_depth": 5, "subsample": 0.9, "colsample_bytree": 0.7,"random_state": 19},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.02, "max_depth": 6, "subsample": 0.85,"colsample_bytree": 0.75,"random_state": 20},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.01, "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.9,"random_state": 21},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.02, "max_depth": 9, "subsample": 0.7, "colsample_bytree": 0.85,"random_state": 22},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.025,"max_depth": 8, "subsample": 0.75,"colsample_bytree": 0.8,"random_state": 23},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.01, "max_depth": 2, "subsample": 0.9, "colsample_bytree": 0.9,"random_state": 24},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.005,"max_depth": 11,"subsample": 0.75,"colsample_bytree": 0.8,"random_state": 35},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.02, "max_depth": 6, "subsample": 0.85,"colsample_bytree": 0.6,"random_state": 26},
    {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.015,"max_depth": 7, "subsample": 0.65,"colsample_bytree": 0.95,"random_state": 27},
]

val_preds = np.zeros(len(y_val))
test_preds = np.zeros(len(X_test))

for i, params in enumerate(param_list):
    print(f"Training model {i+1}/{len(param_list)}...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    val_preds += model.predict(dval, iteration_range=(0, model.best_iteration))
    test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration))

val_preds /= len(param_list)
test_preds /= len(param_list)

print("Validation AUC:", roc_auc_score(y_val, val_preds))
print("Validation Accuracy:", accuracy_score(y_val, (val_preds > 0.5).astype(int)))

submission = pd.DataFrame({
    "id": test["id"],
    "song_popularity": test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")



