import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models.cart import CART
from data.loaders import load_openml_dataset, apply_synthetic_corruption
from experiments.main_benchmark_table4 import PairedTabularDataset, set_seed
import torch.nn.functional as F

def get_counterfactual_delta(model, x_n, x_c, j, num_num, card):
    """Tạo bản sao Counterfactual cho đặc trưng j và tính Delta Xác suất"""
    device = x_n.device
    batch_size = x_n.shape[0]
    
    with torch.no_grad():
        out_base = model(x_n, x_c)
        p_base = F.softmax(out_base[0], dim=1)[:, 1] 
        r_actual = out_base[2] 
        trust_j = r_actual[:, j]

    xn_cf, xc_cf = x_n.clone(), x_c.clone()
    
    if j < num_num:
        xn_cf[:, j] = xn_cf[:, j] + 2.0 
    else:
        cat_idx = j - num_num
        c = card[cat_idx]
        if c > 1:
            xc_cf[:, cat_idx] = (xc_cf[:, cat_idx] + 1) % c

    with torch.no_grad():
        out_cf = model(xn_cf, xc_cf)
        p_cf = F.softmax(out_cf[0], dim=1)[:, 1]
    
    delta_j = torch.abs(p_base - p_cf)
    return trust_j.cpu().numpy(), delta_j.cpu().numpy()

def main():
    EXP_VERSION = "main_results_table4"
    datasets = ['Adult', 'HELOC', 'Higgs', 'Bank', 'Credit', 'Churn', 'Diabetes', 'Magic']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    print("=== MUST 1: COUNTERFACTUAL SENSITIVITY (Q1 KILLER EXPERIMENT) ===")
    
    for d_name in datasets:
        try:
            _, (X_te_n_raw, X_te_c_raw, y_te_raw), card = load_openml_dataset(d_name)
        except: continue
        num_num = X_te_n_raw.shape[1]
        total_feat = num_num + len(card)
        
        for seed in [42, 123, 2024]:
            set_seed(seed)
            ev_n, ev_c, m_ev = apply_synthetic_corruption(X_te_n_raw, X_te_c_raw, card, 0.2)
            dl_eval = DataLoader(PairedTabularDataset(X_te_n_raw, X_te_c_raw, ev_n, ev_c, y_te_raw, m_ev), batch_size=256)
            
            model = CART(num_num, card).to(device)
            ckpt = f"checkpoints/{EXP_VERSION}/seed{seed}/{d_name}_CART-v8-Dynamic.pt"
            if not os.path.exists(ckpt): continue
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            
            all_trust, all_delta = [], []
            for batch in dl_eval:
                xn_cr, xc_cr = batch[2].to(device), batch[3].to(device)
                
                for j in range(total_feat):
                    t_j, d_j = get_counterfactual_delta(model, xn_cr, xc_cr, j, num_num, card)
                    all_trust.extend(t_j)
                    all_delta.extend(d_j)
                    
            df_cf = pd.DataFrame({'Trust': all_trust, 'Delta': all_delta})
            bins = [-np.inf, 
                    df_cf['Trust'].quantile(0.10), 
                    df_cf['Trust'].quantile(0.25), 
                    df_cf['Trust'].quantile(0.50), 
                    df_cf['Trust'].quantile(0.75), 
                    np.inf]
            labels = ['Bottom_10%', '10-25%', '25-50%', '50-75%', 'Top_25%']
            df_cf['Quantile'] = pd.cut(df_cf['Trust'], bins=bins, labels=labels)
            
            mean_delta = df_cf.groupby('Quantile')['Delta'].mean()
            
            res = {'Dataset': d_name, 'Seed': seed}
            for label in labels: res[label] = mean_delta[label]
            results.append(res)
            
            print(f"[{d_name}-{seed}] Deltas -> Bot10%: {res['Bottom_10%']:.4f} | 10-25%: {res['10-25%']:.4f} | ... | Top25%: {res['Top_25%']:.4f}")

    df = pd.DataFrame(results)
    df_agg = df.groupby('Dataset').mean().reset_index()
    df_agg.to_csv("results/counterfactual_sensitivity.csv", index=False)
    print("\nDone!")

if __name__ == "__main__":
    main()