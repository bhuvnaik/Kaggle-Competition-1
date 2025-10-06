# IISc UMC-301: Song Popularity Prediction

Two distinct ensemble strategies were explored for predicting song popularity, 
submission-1.py — A spectral-statistical ensemble using handcrafted rhythmic, harmonic, and acoustic interactions, aligning interpretability with variance-aware boosting.
submission-2.py — A geometric-latent ensemble combining PCA compression and Node2Vec-style embeddings to uncover relational structure beneath the audio feature space.

# Submission Scores

| File | Public AUC | Private AUC |
|:------|:-----------:|:------------:|
| **submission-1.py** | **0.58937** | **0.56577** |
| **submission-2.py** | **0.58409** | **0.56213** |

Both submissions were cross-validated using **Stratified K-Fold** and evaluated on **ROC-AUC**.  
While the spectral ensemble achieved higher leaderboard scores, the embedding-based model provided a more generalized latent structure.

**Bhuvan Naik**  
*IISc UMC-301 · Kaggle Competition-1 2025*
