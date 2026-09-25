import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings("ignore")

class TabularDataset(Dataset):
    def __init__(self, x_num, x_cat, y, mask=None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.float32) if mask is not None else torch.ones_like(self.x_num)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx], self.mask[idx]

def load_openml_dataset(dataset_name, data_dir='./data'):
    """Tải và Cache 8 Benchmark Datasets."""
    os.makedirs(data_dir, exist_ok=True)
    
    openml_mapping = {
        'Adult': 'adult',
        'HELOC': 'heloc',
        'Higgs': 23512, # ID bản subset của Higgs
        'Bank': 'bank-marketing',
        'Credit': 'default-of-credit-card-clients',
        'Churn': 'churn',
        'Diabetes': 'diabetes',
        'Magic': 'MagicTelescope'
    }
    
    fetch_id = openml_mapping.get(dataset_name, dataset_name)
    
    try:
        if isinstance(fetch_id, int):
            data = fetch_openml(data_id=fetch_id, as_frame=True, parser='auto', data_home=data_dir)
        else:
            data = fetch_openml(name=fetch_id, version=1, as_frame=True, parser='auto', data_home=data_dir)
    except Exception:
        data = fetch_openml(name=fetch_id, version=2, as_frame=True, parser='auto', data_home=data_dir)
        
    df = data.frame.dropna()
    
    # Giới hạn kích thước để train nhanh
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    target_col = data.target_names[0]
    cat_cols = df.select_dtypes(include=['category', 'object', 'bool']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if target_col in cat_cols: cat_cols.remove(target_col)
    if target_col in num_cols: num_cols.remove(target_col)

    y = LabelEncoder().fit_transform(df[target_col])
    X_num = df[num_cols].values if len(num_cols) > 0 else np.empty((len(df), 0))
    
    X_cat = np.zeros((len(df), len(cat_cols)), dtype=int)
    cardinalities = []
    for i, col in enumerate(cat_cols):
        le = LabelEncoder()
        X_cat[:, i] = le.fit_transform(df[col].astype(str))
        cardinalities.append(len(le.classes_))

    X_n_tmp, X_n_te, X_c_tmp, X_c_te, y_tmp, y_te = train_test_split(X_num, X_cat, y, test_size=0.15, random_state=42)
    X_n_tr, X_n_v, X_c_tr, X_c_v, y_tr, y_v = train_test_split(X_n_tmp, X_c_tmp, y_tmp, test_size=0.176, random_state=42)

    scaler = StandardScaler()
    if X_n_tr.shape[1] > 0:
        X_n_tr = scaler.fit_transform(X_n_tr)
        X_n_te = scaler.transform(X_n_te)

    return (X_n_tr, X_c_tr, y_tr), (X_n_te, X_c_te, y_te), cardinalities

def apply_synthetic_corruption(X_num, X_cat, cardinalities, corruption_rate=0.2):
    X_num_c, X_cat_c = X_num.copy(), X_cat.copy()
    mask_num, mask_cat = np.ones_like(X_num, dtype=np.float32), np.ones_like(X_cat, dtype=np.float32)
    
    if X_num.shape[1] > 0:
        corrupt_idx_num = np.random.rand(*X_num.shape) < corruption_rate
        noise = np.random.randn(*X_num.shape) * 2.0
        X_num_c[corrupt_idx_num] += noise[corrupt_idx_num]
        mask_num[corrupt_idx_num] = 0.0 
    
    if X_cat.shape[1] > 0:
        corrupt_idx_cat = np.random.rand(*X_cat.shape) < corruption_rate
        effective_mask = np.zeros_like(corrupt_idx_cat, dtype=bool)
        
        for j in range(X_cat.shape[1]):
            # Chỉ nhiễu nếu có nhiều hơn 1 class (Tránh đánh nhầm mask=0 cho cột c==1)
            if cardinalities[j] > 1: 
                mask_j = corrupt_idx_cat[:, j]
                effective_mask[:, j] = mask_j
                
                orig_vals = X_cat[mask_j, j]
                rand_vals = np.random.randint(0, cardinalities[j], size=mask_j.sum())
                
                # Đảm bảo giá trị nhiễu PHẢI KHÁC giá trị gốc
                same_mask = (rand_vals == orig_vals)
                rand_vals[same_mask] = (rand_vals[same_mask] + 1) % cardinalities[j]
                
                X_cat_c[mask_j, j] = rand_vals
                
        # Chỉ đánh mask=0 cho những ô thực sự đã bị đổi giá trị
        mask_cat[effective_mask] = 0.0
    
    return X_num_c, X_cat_c, np.concatenate([mask_num, mask_cat], axis=1)

def apply_contextual_corruption(X_num, X_cat, cardinalities, corruption_rate=0.2):
    """Contextual tổng quát: Dựa vào cột 0 để bơm nhiễu cột cuối."""
    X_num_c, X_cat_c = X_num.copy(), X_cat.copy()
    mask_num, mask_cat = np.ones_like(X_num, dtype=np.float32), np.ones_like(X_cat, dtype=np.float32)
    
    if X_num.shape[1] >= 2:
        anchor_col = 0
        target_col = X_num.shape[1] - 1
        
        # Ngữ cảnh: Nếu cột anchor mang giá trị âm (sau scale)
        context_mask = X_num[:, anchor_col] < 0 
        corrupt_prob = np.random.rand(len(X_num))
        corrupt_idx = context_mask & (corrupt_prob < (corruption_rate * 2)) 
        
        X_num_c[corrupt_idx, target_col] += np.random.uniform(5.0, 10.0, size=corrupt_idx.sum())
        mask_num[corrupt_idx, target_col] = 0.0
    else:
        # Fallback nếu dataset không đủ cột numerical
        return apply_synthetic_corruption(X_num, X_cat, cardinalities, corruption_rate)
        
    return X_num_c, X_cat_c, np.concatenate([mask_num, mask_cat], axis=1)