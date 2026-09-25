import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

class XGBoostBaseline:
    def __init__(self, random_state=42):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            tree_method='hist',
            random_state=random_state,
            n_jobs=-1
        )
        
    def _prepare_data(self, X_num, X_cat):
        """Nối mảng số thực và phân loại để nhét vào Tree"""
        if X_cat.shape[1] > 0:
            return np.concatenate([X_num, X_cat], axis=1)
        return X_num

    def fit(self, X_num, X_cat, y):
        X = self._prepare_data(X_num, X_cat)
        self.model.fit(X, y)

    def evaluate(self, X_num, X_cat, y):
        X = self._prepare_data(X_num, X_cat)
        preds = self.model.predict(X)
        return accuracy_score(y, preds) * 100