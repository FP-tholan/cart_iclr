import os
import torch
import joblib
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

from models.cart import CART
from baselines.quail_base import QuAILBaseline
from baselines.mlp_base import VanillaMLP
from baselines.ft_transformer import FTTransformer
from baselines.xgboost_base import XGBoostBaseline
from baselines.catboost_base import CatBoostBaseline
from data.loaders import load_openml_dataset
from experiments.main_benchmark_table4 import PairedTabularDataset, set_seed

def apply_unseen_magnitude(X_num, X_cat, corruption_rate=0.2):
    X_n_c = X_num.copy()
    if X_n_c.shape[1] > 0:
        mask = np.random.rand(*X_n_c.shape) < corruption_rate
        noise = np.random.uniform(3.0, 5.0, size=X_n_c.shape)
        sign = np.random.choice([-1, 1], size=X_n_c.shape)
        X_n_c[mask] += (noise * sign)[mask]
    return X_n_c, X_cat.copy()

def apply_feature_shift(X_num, X_cat):
    X_n_c = X_num.copy()
    if X_n_c.shape[1] > 0:
        target_col = np.random.randint(0, X_n_c.shape[1])
        X_n_c[:, target_col] += np.random.uniform(2.0, 4.0)
    return X_n_c, X_cat.copy()

def eval_tree(model_obj, x_n, x_c, y, card):
    if len(card) > 0:
        X_ev = pd.concat([pd.DataFrame(x_n, dtype=float), pd.DataFrame(x_c, dtype=int)], axis=1)
        X_ev = X_ev.astype(str).values
    else:
        X_ev = x_n
    preds = model_obj.model.predict(X_ev)
    return accuracy_score(y, preds) * 100

def eval_nn(model_obj, loader, device):
    preds, y_all = [], []
    with torch.no_grad():
        for batch in loader:
            xn, xc, yt = batch[2].to(device), batch[3].to(device), batch[4].to(device)
            out = model_obj(xn, xc)
            logits = out[0] if isinstance(out, tuple) else out
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            y_all.extend(yt.cpu().numpy())
    return accuracy_score(y_all, preds) * 100

def main():
    EXP_VERSION = "main_results_table4"
    datasets = ['Adult', 'HELOC', 'Higgs', 'Bank', 'Credit', 'Churn', 'Diabetes', 'Magic']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results_mag, results_fs = [], []
    print("=== UNSEEN CORRUPTION TRÊN CÁC MODEL 80-EPOCH ===")
    
    for d_name in datasets:
        try:
            _, (X_te_n, X_te_c, y_te), card = load_openml_dataset(d_name)
        except: continue
        num_num = X_te_n.shape[1]
        
        for seed in [42, 123, 2024]:
            set_seed(seed)
            X_mag_n, X_mag_c = apply_unseen_magnitude(X_te_n, X_te_c, 0.2)
            X_fs_n, X_fs_c = apply_feature_shift(X_te_n, X_te_c)
            
            dummy_m = np.ones((X_te_n.shape[0], num_num + len(card)), dtype=np.float32)
            dl_mag = DataLoader(PairedTabularDataset(X_te_n, X_te_c, X_mag_n, X_mag_c, y_te, dummy_m), batch_size=256)
            dl_fs = DataLoader(PairedTabularDataset(X_te_n, X_te_c, X_fs_n, X_fs_c, y_te, dummy_m), batch_size=256)
            
            res_m = {'Dataset': d_name, 'Seed': seed}
            res_f = {'Dataset': d_name, 'Seed': seed}
            
            models = {
                'MLP-Aug-80ep': (VanillaMLP(num_num, card), 'nn_base_80'),
                'FT-Transformer-Aug-80ep': (FTTransformer(num_num, card), 'nn_base_80'),
                'QuAIL-Aug-80ep': (QuAILBaseline(num_num, card), 'nn_base_80'),
                'XGBoost-Aug': (XGBoostBaseline(), 'tree'),
                'CatBoost-Aug': (CatBoostBaseline(), 'tree'),
                'CART-v8-Dynamic': (CART(num_num, card), 'nn_cart_dynamic')
            }
            
            ckpt_dir = f"checkpoints/{EXP_VERSION}/seed{seed}"
            for m_name, (m_obj, m_type) in models.items():
                
                safe_m_name = m_name.replace(' ', '_').replace('(', '').replace(')', '')
                ckpt_path = f"{ckpt_dir}/{d_name}_{safe_m_name}.pt"
                
                if not os.path.exists(ckpt_path): 
                    print(f"CẢNH BÁO: Không tìm thấy {ckpt_path}")
                    continue
                
                if m_type == 'tree':
                    m_obj.model = joblib.load(ckpt_path)
                    res_m[m_name] = eval_tree(m_obj, X_mag_n, X_mag_c, y_te, card)
                    res_f[m_name] = eval_tree(m_obj, X_fs_n, X_fs_c, y_te, card)
                else:
                    m_obj.load_state_dict(torch.load(ckpt_path, map_location=device))
                    m_obj.to(device).eval()
                    res_m[m_name] = eval_nn(m_obj, dl_mag, device)
                    res_f[m_name] = eval_nn(m_obj, dl_fs, device)
            
            results_mag.append(res_m)
            results_fs.append(res_f)
            
            cart_mag = res_m.get('CART-v8-Dynamic', 0)
            cart_fs = res_f.get('CART-v8-Dynamic', 0)
            print(f"[{d_name} - Seed {seed}] CART Mag: {cart_mag:.1f} | CART FS: {cart_fs:.1f}")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results_mag).groupby('Dataset').mean().to_csv("results/unseen_magnitude_baselines.csv")
    pd.DataFrame(results_fs).groupby('Dataset').mean().to_csv("results/unseen_featshift_baselines.csv")
    print("\n Done!")

if __name__ == "__main__":
    main()