import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

class CatBoostBaseline:
    def __init__(self, random_state=42):
        self.model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.1,
            depth=6,
            verbose=0,
            random_state=random_state,
            thread_count=-1
        )
        
    def _prepare_data(self, X_num, X_cat):
        if X_cat.shape[1] > 0:
            df_num = pd.DataFrame(X_num)
            df_cat = pd.DataFrame(X_cat, dtype=int)
            X = pd.concat([df_num, df_cat], axis=1)
            X.columns = range(X.shape[1])  
            cat_features = list(range(X_num.shape[1], X.shape[1]))
            return X, cat_features
        return pd.DataFrame(X_num), []

    def fit(self, X_num, X_cat, y):
        X, cat_features = self._prepare_data(X_num, X_cat)
        if len(cat_features) > 0:
            self.model.fit(X, y, cat_features=cat_features)
        else:
            self.model.fit(X, y)

    def evaluate(self, X_num, X_cat, y):
        X, _ = self._prepare_data(X_num, X_cat)
        preds = self.model.predict(X)
        return accuracy_score(y, preds) * 100