import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy.stats
from torch.utils.data import DataLoader
from models.cart import CART
from counterfactual.perturb import generate_single_cell_counterfactual
from data.loaders import load_openml_dataset
from experiments.main_benchmark_table4 import PairedTabularDataset, set_seed

def main():
    EXP_VERSION = "main_results_table1"
    datasets = ['Adult', 'HELOC', 'Bank', 'Credit', 'Churn']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    print("=== COUNTERFACTUAL STABILITY VALIDATION ===")
    
    for d_name in datasets:
        try:
            _, (X_te_n_raw, X_te_c_raw, y_te_raw), card = load_openml_dataset(d_name)
        except: continue
        num_num = X_te_n_raw.shape[1]
        total_feat = num_num + len(card)
        
        for seed in [42]: 
            set_seed(seed)
            limit = min(1000, X_te_n_raw.shape[0])
            xn, xc, y_t = X_te_n_raw[:limit], X_te_c_raw[:limit], y_te_raw[:limit]
            
            model = CART(num_num, card).to(device)
            ckpt_path = f"checkpoints/{EXP_VERSION}/seed{seed}/{d_name}_CART-v8-Dynamic.pt"
            if not os.path.exists(ckpt_path): continue
            
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            
            xn_t, xc_t = torch.tensor(xn, dtype=torch.float32).to(device), torch.tensor(xc, dtype=torch.long).to(device)
            
            with torch.no_grad():
                logits_base, _, trust_base, _, _, _, _ = model(xn_t, xc_t)
                prob_base = F.softmax(logits_base, dim=1)[:, 1] 
                
                all_trusts = []
                all_minus_deltas = []
                
                for j in range(total_feat):
                    xn_cf, xc_cf = generate_single_cell_counterfactual(xn_t, xc_t, card, j)
                    logits_cf = model(xn_cf, xc_cf)[0]
                    prob_cf = F.softmax(logits_cf, dim=1)[:, 1]
                    
                    delta = torch.abs(prob_base - prob_cf)
                    
                    all_trusts.extend(trust_base[:, j].cpu().numpy())
                    all_minus_deltas.extend((-delta).cpu().numpy())
            
            spearman_corr, p_value = scipy.stats.spearmanr(all_trusts, all_minus_deltas)
            results.append({'Dataset': d_name, 'Spearman_Corr': spearman_corr, 'P_Value': p_value})
            print(f"[{d_name}] Tương quan Trust và (-Delta): {spearman_corr:.4f} (p={p_value:.4e})")

    pd.DataFrame(results).to_csv("results/counterfactual_correlation.csv", index=False)
    print("Trust cao -> Perturb càng ít.")

if __name__ == "__main__":
    main()