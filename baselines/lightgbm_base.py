import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

class LightGBMBaseline:
    def __init__(self, random_state=42):
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
        
    def _prepare_data(self, X_num, X_cat):
        if X_cat.shape[1] > 0:
            df_num = pd.DataFrame(X_num)
            df_cat = pd.DataFrame(X_cat, dtype='category')
            X = pd.concat([df_num, df_cat], axis=1)
            X.columns = [str(i) for i in range(X.shape[1])]
            return X
        return pd.DataFrame(X_num)

    def fit(self, X_num, X_cat, y):
        X = self._prepare_data(X_num, X_cat)
        self.model.fit(X, y)

    def evaluate(self, X_num, X_cat, y):
        X = self._prepare_data(X_num, X_cat)
        preds = self.model.predict(X)
        return accuracy_score(y, preds) * 100